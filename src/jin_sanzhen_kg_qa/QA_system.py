

from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache, wraps
import os
import json
import logging
import time
import difflib

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from py2neo import Graph
from openai import OpenAI

# ==========================
# 日志
# ==========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jzs_qa")

# ==========================
# DashScope (Qwen) 兼容 OpenAI SDK
# ==========================
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
if not DASHSCOPE_API_KEY:
    raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY")

client = OpenAI(
    base_url=DASHSCOPE_BASE_URL,
    api_key=DASHSCOPE_API_KEY,
)

QWEN_MODEL_PARSE = os.getenv("QWEN_MODEL_PARSE", "qwen3-max")
QWEN_MODEL_ANSWER = os.getenv("QWEN_MODEL_ANSWER", "qwen3-max")

# ==========================
# Neo4j 配置
# ==========================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Jacky@0906")

graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ==========================
# Pydantic 模型
# ==========================
class HistoryItem(BaseModel):
    role: str
    content: str


class QAOptions(BaseModel):
    max_plans: Optional[int] = 10
    language: Optional[str] = "zh"


class QARequest(BaseModel):
    question: str
    history: Optional[List[HistoryItem]] = None
    options: Optional[QAOptions] = None
    use_kg: Optional[bool] = True


class QAResponse(BaseModel):
    answer: str
    query_type: Optional[str] = None
    entities: Optional[Dict[str, List[str]]] = None
    cypher: Optional[str] = None
    records: Optional[List[Dict[str, Any]]] = None
    entity_suggest: Optional[Dict[str, Any]] = None
    answer_source: Optional[str] = None  # "kg_hybrid" | "llm_only"


# ==========================
# 带 TTL 的 lru_cache
# ==========================
def timed_lru_cache(seconds: int = 300, maxsize: int = 1):
    def decorator(func):
        cached_func = lru_cache(maxsize=maxsize)(func)
        cached_func._expiry = 0.0

        @wraps(func)
        def wrapper(*args, **kwargs):
            if time.time() > cached_func._expiry:
                cached_func.cache_clear()
                cached_func._expiry = time.time() + seconds
            return cached_func(*args, **kwargs)

        wrapper.cache_clear = cached_func.cache_clear
        return wrapper

    return decorator


# ==========================
# 实体缓存
# ==========================
@timed_lru_cache(seconds=600)
def get_all_diseases() -> List[str]:
    data = graph.run("MATCH (d:Disease) RETURN DISTINCT d.name AS name").data()
    return [r["name"] for r in data if r.get("name")]


@timed_lru_cache(seconds=600)
def get_all_combos() -> List[str]:
    data = graph.run("MATCH (c:AcupointCombo) RETURN DISTINCT c.name AS name").data()
    return [r["name"] for r in data if r.get("name")]


@timed_lru_cache(seconds=600)
def get_all_points() -> List[str]:
    data = graph.run("MATCH (a:Acupoint) RETURN DISTINCT a.name AS name").data()
    return [r["name"] for r in data if r.get("name")]


# ==========================
# difflib 兜底（最后一道防线）
# ==========================
def _best_match(
    name: str,
    candidates: List[str],
    cutoff: float = 0.5,
    top_n: int = 5,
) -> Tuple[Optional[str], List[str]]:
    if not name or not candidates:
        return None, []
    res = difflib.get_close_matches(name, candidates, n=top_n, cutoff=cutoff)
    if not res:
        return None, []
    best = res[0]
    ratio = difflib.SequenceMatcher(None, name, best).ratio()
    if ratio < cutoff:
        return None, res
    return best, res


# ==========================
# ★ LLM 语义实体对齐
# ==========================
def _llm_entity_align(
    unmatched: Dict[str, List[str]],
    candidates: Dict[str, List[str]],
) -> Dict[str, Dict[str, Optional[str]]]:
    has_work = any(v for v in unmatched.values())
    if not has_work:
        return {"diseases": {}, "combos": {}, "points": {}}

    system_prompt = """
你是一个中医针灸领域的术语对齐助手。

用户在提问中使用的疾病名、靳三针组合名、穴位名，可能与知识图谱中的标准名称不一致
（常见情况：同义词、简称、别名、俗称、古今异名等）。

你的任务：将每个"用户名称"匹配到"候选列表"中语义最接近的标准名称。

返回严格 JSON，格式：
{
  "diseases": {"用户名称A": "标准名称或null", ...},
  "combos":   {"用户名称B": "标准名称或null", ...},
  "points":   {"用户名称C": "标准名称或null", ...}
}

规则：
1. 只能返回候选列表中确实存在的名称，不可编造。
2. 基于中医 / 西医学语义判断，例如"中风"="脑卒中后遗症"，"面瘫"="周围性面瘫"。
3. 如果候选列表中确实没有语义对应的名称，返回 null。
4. 只返回 JSON，不加任何注释或解释文字。
"""

    payload: Dict[str, Any] = {}
    for entity_type in ("diseases", "combos", "points"):
        names = unmatched.get(entity_type, [])
        if names:
            payload[entity_type] = {
                "to_match": names,
                "candidates": candidates.get(entity_type, []),
            }

    user_content = json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        content = _chat_completion(
            model=QWEN_MODEL_PARSE,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_content},
            ],
            temperature=0.05,
        )
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        result = json.loads(text)
        result.setdefault("diseases", {})
        result.setdefault("combos", {})
        result.setdefault("points", {})
        logger.info(f"LLM 实体对齐结果: {result}")
        return result

    except Exception:
        logger.exception("LLM 实体对齐调用失败，返回空映射")
        return {"diseases": {}, "combos": {}, "points": {}}


# ==========================
# ★ normalize_entities：精确匹配 → LLM 对齐 → difflib 兜底
# ==========================
def normalize_entities(
    parsed: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    normalized = parsed.copy()
    suggest: Dict[str, Any] = {"unmatched": [], "candidates": {}}

    all_diseases = get_all_diseases()
    all_combos = get_all_combos()
    all_points = get_all_points()

    diseases_set = set(all_diseases)
    combos_set = set(all_combos)
    points_set = set(all_points)

    unmatched: Dict[str, List[str]] = {"diseases": [], "combos": [], "points": []}

    for d in parsed.get("diseases") or []:
        if d not in diseases_set:
            unmatched["diseases"].append(d)

    for c in parsed.get("combos") or []:
        if c not in combos_set:
            unmatched["combos"].append(c)

    for p in parsed.get("points") or []:
        if p not in points_set:
            unmatched["points"].append(p)

    llm_mapping: Dict[str, Dict[str, Optional[str]]] = {
        "diseases": {},
        "combos": {},
        "points": {},
    }
    if any(unmatched.values()):
        logger.info(f"发现未精确匹配的实体，调用 LLM 对齐: {unmatched}")
        llm_mapping = _llm_entity_align(
            unmatched=unmatched,
            candidates={
                "diseases": all_diseases,
                "combos": all_combos,
                "points": all_points,
            },
        )

    norm_d = []
    for d in parsed.get("diseases") or []:
        if d in diseases_set:
            norm_d.append(d)
        else:
            mapped = llm_mapping.get("diseases", {}).get(d)
            if mapped and mapped in diseases_set:
                norm_d.append(mapped)
                logger.info(f"[LLM 对齐] Disease: '{d}' → '{mapped}'")
            else:
                best, cands = _best_match(d, all_diseases, cutoff=0.5)
                if best:
                    norm_d.append(best)
                    logger.info(f"[difflib 兜底] Disease: '{d}' → '{best}'")
                else:
                    norm_d.append(d)
                    suggest["unmatched"].append({"type": "disease", "name": d})
                    if cands:
                        suggest["candidates"].setdefault(d, {})["diseases"] = cands
    normalized["diseases"] = norm_d

    norm_c = []
    for c in parsed.get("combos") or []:
        if c in combos_set:
            norm_c.append(c)
        else:
            mapped = llm_mapping.get("combos", {}).get(c)
            if mapped and mapped in combos_set:
                norm_c.append(mapped)
                logger.info(f"[LLM 对齐] Combo: '{c}' → '{mapped}'")
            else:
                best, cands = _best_match(c, all_combos, cutoff=0.5)
                if best:
                    norm_c.append(best)
                    logger.info(f"[difflib 兜底] Combo: '{c}' → '{best}'")
                else:
                    norm_c.append(c)
                    suggest["unmatched"].append({"type": "combo", "name": c})
                    if cands:
                        suggest["candidates"].setdefault(c, {})["combos"] = cands
    normalized["combos"] = norm_c

    norm_p = []
    for p in parsed.get("points") or []:
        if p in points_set:
            norm_p.append(p)
        else:
            mapped = llm_mapping.get("points", {}).get(p)
            if mapped and mapped in points_set:
                norm_p.append(mapped)
                logger.info(f"[LLM 对齐] Point: '{p}' → '{mapped}'")
            else:
                best, cands = _best_match(p, all_points, cutoff=0.5)
                if best:
                    norm_p.append(best)
                    logger.info(f"[difflib 兜底] Point: '{p}' → '{best}'")
                else:
                    norm_p.append(p)
                    suggest["unmatched"].append({"type": "point", "name": p})
                    if cands:
                        suggest["candidates"].setdefault(p, {})["points"] = cands
    normalized["points"] = norm_p

    if not suggest["unmatched"] and not suggest["candidates"]:
        suggest = {}

    return normalized, suggest


# ==========================
# Records 清洗
# ==========================
def _clean_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for r in records:
        row = {}
        for k, v in r.items():
            if v is None:
                continue
            if isinstance(v, bool):
                row[k] = v
                continue
            if isinstance(v, list):
                v = [item for item in v if item is not None]
                if v and isinstance(v[0], dict):
                    v = [item for item in v if item.get("name")]
                if not v:
                    continue
            if isinstance(v, str) and not v.strip():
                continue
            row[k] = v
        if row:
            cleaned.append(row)
    return cleaned


# ==========================
# LLM 调用
# ==========================
def _chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def call_llm_for_answer_no_kg(
    question: str,
    query_type: str,
    entities: Dict[str, List[str]],
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    if query_type == "unknown":
        return (
            "当前无法从问题中提取出明确的疾病、靳三针组合或穴位信息，"
            "建议你补充更具体的描述，例如“脑卒中后遗症常用哪些靳三针方案？”。"
        )

    system_prompt = """
你是一个靳三针针灸领域的学术型问答助手。
当前模式为"仅 LLM（不依赖知识图谱）"，
请基于你已有的通用医学知识和中医针灸知识，
围绕用户问题进行回答，重点说明靳三针相关内容。

要求：
- 用中文专业、客观地回答，尽量结构清晰。
- 可以结合你已知的靳三针理论、常见取穴组合、常见适应证和疗效结论进行总结。
- 不能声称"来自本系统图谱"，也不要编造具体文献标题或编号。
- 不要给出具体处方或个体化诊疗建议。
- 回答最后加一句提醒：
  "以上内容主要基于通用中医针灸与靳三针相关知识，供教学与科研参考，不构成个体化诊疗建议。"
"""
    user_payload = {
        "question": question,
        "query_type": query_type,
        "entities": entities,
    }

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt.strip()},
    ]
    if history:
        for h in history:
            if h.get("role") in ("user", "assistant"):
                messages.append({"role": h["role"], "content": h.get("content", "")})
    messages.append({
        "role": "user",
        "content": (
            "下面是本次查询的结构化解析数据，请结合这些信息并依靠你自身知识回答：\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2)
        ),
    })

    try:
        return _chat_completion(model=QWEN_MODEL_ANSWER, messages=messages, temperature=0.5)
    except Exception:
        logger.exception("LLM 生成答案失败（no KG 模式）")
        return "当前在未调用知识图谱的前提下生成答案失败，建议稍后重试。"


def call_llm_for_parse(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    system_prompt = """
你是一个靳三针针灸知识图谱问答系统的问题解析助手。
请把用户问题解析为一个 JSON，结构如下（必须严格 JSON）：

{
  "query_type": "<string>",
  "diseases": ["..."],
  "combos": ["..."],
  "points": ["..."]
}

query_type 必须从下列值中选一个：
- "disease_to_plans"          ：问"某个疾病有哪些靳三针治疗方案"
- "combo_to_diseases"         ：问"某个靳三针组合能治什么病"
- "disease_combo_to_effect"   ：问"某个组合治疗某病的疗效如何"
- "point_to_combos"           ：问"某个穴位参与哪些靳三针组合/用于哪些病"
- "disease_to_point_summary"  ：问"某病靳三针常用主穴/配穴总结"
- "combo_to_points"           ：问"某个靳三针组合包含哪些穴位/子组合"
- "disease_compare_plans"     ：问"某病不同方案哪个效果好 / 方案对比"
- "combo_detail"              ：问"某个或多个靳三针组合的详细信息、穴位构成、适应证"，也包括"两个组合有什么区别/异同/分别用于什么病"
- "multi_disease_common"      ：问"两种或多种疾病有哪些共用的靳三针组合或穴位"
- "unknown"                   ：无法归入以上任何一类

说明：
- diseases：提取文本中的疾病名称（用户原文中的说法即可，后续系统会自动对齐标准名称）。
- combos：提取靳三针"组合名称"，如"颞三针"、"智三针"等。
- points：提取单个标准穴位名，如"合谷"、"足三里"。
- 如果用户使用了"X（Y）"或"X(Y)"的写法，请将 X 作为实体名称提取，括号内的 Y 视为别名，不要单独提取 Y。
  例如："四神针（四神聪）"应提取为 combos: ["四神针"]，而不是 points: ["四神聪"]。
- 带有"针"字后缀的名称（如"四神针""颞三针""智三针"）通常是靳三针组合名，应放入 combos 而非 points。

★ 多轮对话指代消解：
- 如果用户使用了代词（如"它""这个""那个""该组合""上面那个病"等），
  或者问题中省略了主语（如"包含哪些穴位？""怎么进针？"），
  请根据对话历史（之前的 user/assistant 消息）推断所指的具体实体，
  并将推断出的实体填入对应的 diseases / combos / points 字段。
- 例如：上一轮问了"颞三针能治什么病？"，本轮问"那它包含哪些穴位？"，
  则应解析为 combos: ["颞三针"], query_type: "combo_to_points"。
- 如果对话历史为空且无法推断指代，才归为 "unknown"。

必须：
1）只返回 JSON，不加任何注释文字；
2）所有字段都要出现，即使为空也要给空数组。
"""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt.strip()}
    ]
    if history:
        for h in history:
            if h.get("role") in ("user", "assistant"):
                messages.append({"role": h["role"], "content": h.get("content", "")})
    messages.append({"role": "user", "content": question})

    try:
        content = _chat_completion(model=QWEN_MODEL_PARSE, messages=messages, temperature=0.1)
        logger.info(f"LLM raw parse: {content}")
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        parsed.setdefault("query_type", "disease_to_plans")
        parsed.setdefault("diseases", [])
        parsed.setdefault("combos", [])
        parsed.setdefault("points", [])
        return parsed
    except Exception:
        logger.exception("LLM 解析失败，使用简单规则兜底")
        q = question.strip()
        if "颞三针" in q and ("什么病" in q or "哪些病" in q or "适应证" in q):
            return {"query_type": "combo_to_diseases", "diseases": [], "combos": ["颞三针"], "points": []}
        if "脑卒中" in q:
            return {"query_type": "disease_to_plans", "diseases": ["脑卒中后遗症"], "combos": [], "points": []}
        return {"query_type": "unknown", "diseases": [], "combos": [], "points": []}


# ==============================================================
# ★★★ 核心改动：分层可信度生成（Tiered Confidence Generation）
# ==============================================================
def call_llm_for_answer(
    question: str,
    records: List[Dict[str, Any]],
    query_type: str,
    entities: Dict[str, List[str]],
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    v2.3 分层可信度生成：
      第一层 📌 严格基于 KG records（高可信度）
      第二层 💡 LLM 通用知识补充（较低可信度，禁止编造穴位组成）
    """
    if query_type == "unknown":
        return (
            "当前无法从问题中提取出明确的疾病、靳三针组合或穴位信息，"
            "建议你补充更具体的描述，例如”脑卒中后遗症常用哪些靳三针方案？“。"
        )

    has_kg_records = bool(records)

    system_prompt = """
你是一个靳三针知识图谱问答助手，采用"分层可信度生成策略"回答用户问题。
你的回答必须严格分为以下两个层级，并使用指定的标注格式。

================================================================
【第一层：基于知识图谱的回答】— 高可信度
================================================================
- 严格且仅基于下方 records（JSON 列表）中的信息回答。
- 不得添加 records 里没有出现的穴位名称、组合名称、疾病名称或它们之间的具体关系。
- 如果 records 不为空，请充分、完整地利用其中的信息来回答用户问题。
- 如果 records 为空，或者 records 中的信息不足以回答用户问题的某些子问题，
  请在第一层中对缺失部分明确说明"当前知识图谱中暂未收录该部分信息"，
  然后将这些子问题留给第二层补充。
- 输出格式——在该层内容最前面写：

  📌 **以下内容基于靳三针知识图谱检索结果（高可信度）：**

不同 query_type 时第一层的侧重点：
- disease_to_plans：概括主穴组合、辅助穴位、电针/艾灸/药物配合、疗程和疗效结论等。
- combo_to_diseases：总结该组合主要用于哪些疾病或病证。
- disease_combo_to_effect：说明研究设计、疗程、主要疗效指标。
- point_to_combos：总结该穴位参与了哪些组合，这些组合用于哪些疾病。
- disease_to_point_summary：列出某病常用主穴、配穴。
- combo_to_points：总结该组合由哪些标准穴位、局部点和子组合构成。
- disease_compare_plans：对比同一疾病下不同方案的疗效差异。
- combo_detail：全面展示该组合的穴位构成、刺法、适应证等完整信息，若 records 中包含多个组合，应对比说明各自穴位构成与适应证的异同。
- multi_disease_common：列出多种疾病共用的组合或穴位。

================================================================
【第二层：基于通用知识的补充】— 较低可信度
================================================================
- 针对第一层无法覆盖的子问题，你可以基于你的中医针灸通用知识进行补充。
- 输出格式——在该层内容最前面写：

  💡 **以下内容基于通用中医针灸知识补充，未经本知识图谱验证（较低可信度）：**

- ⛔ 第二层严格禁止的内容：
  1. 编造靳三针特有的穴位组合名称（如"XX三针"）及其具体穴位构成；
  2. 编造具体穴位处方、针刺深度、留针时间等参数；
  3. 虚构文献标题、作者姓名或编号；
  4. 声称某信息"来自本知识图谱"或"据图谱记载"。

- ✅ 第二层允许补充的内容范围：
  1. 中医辨证思路与病机分析（如"该病多属心肾不交、肝郁脾虚"等）；
  2. 一般性的治疗原则与配伍思路（如"多以调神益智、醒脑开窍为治则"）；
  3. 临床疗效的概括性描述（如"多项研究表明针灸联合康复训练优于单纯康复"）；
  4. 常用评估量表或疗效指标的简介（如 CARS、ABC、GMFM 等）；
  5. 安全性、注意事项、疗程建议的一般性说明；
  6. 该疾病或组合在中医理论中的定位与归属。

- 如果第一层已经完整回答了用户的所有子问题，则不需要生成第二层——
  不要为了凑字数而强行补充。

================================================================
【结尾提醒】
================================================================
回答最后必须加上这段话（原文照抄）：

"以上内容仅供教学与科研参考，不构成个体化诊疗建议。其中标注为'知识图谱检索结果'的部分可信度较高，标注为'通用知识补充'的部分请结合权威教材进一步查证。"
"""

    # 当 records 为空时，给 LLM 明确的提示
    records_for_prompt: Any
    if has_kg_records:
        records_for_prompt = records
    else:
        records_for_prompt = "（本次查询在知识图谱中未检索到相关记录，第一层应说明此情况，请在第二层进行补充。）"

    user_payload = {
        "question": question,
        "query_type": query_type,
        "entities": entities,
        "records": records_for_prompt,
    }

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt.strip()},
    ]
    if history:
        for h in history:
            if h.get("role") in ("user", "assistant"):
                messages.append({"role": h["role"], "content": h.get("content", "")})
    messages.append({
        "role": "user",
        "content": (
            "下面是本次查询的完整数据，请严格按照分层可信度策略作答：\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2)
        ),
    })

    try:
        return _chat_completion(model=QWEN_MODEL_ANSWER, messages=messages, temperature=0.3)
    except Exception:
        logger.exception("LLM 生成答案失败")
        if has_kg_records:
            return (
                "系统已成功从知识图谱中检索到相关靳三针方案，"
                "但在生成自然语言总结时发生错误。建议直接查看原始 records 数据。"
            )
        else:
            return (
                "在目前已构建的靳三针知识图谱中，暂未检索到与该问题直接对应的结构化记录，"
                "且自然语言生成环节也发生了错误。"
                "请结合权威教材和临床指南进一步查证。"
            )


# ==========================
# ★ query_type 自动降级/纠正
# ==========================
def _adjust_query_type(
    query_type: str,
    entities: Dict[str, List[str]],
) -> str:
    diseases = entities.get("diseases") or []
    combos = entities.get("combos") or []
    points = entities.get("points") or []

    original = query_type

    if query_type in ("disease_to_plans", "disease_to_point_summary", "disease_compare_plans"):
        if not diseases:
            if combos:
                query_type = "combo_to_diseases"
            elif points:
                query_type = "point_to_combos"
            else:
                query_type = "unknown"

    elif query_type in ("combo_to_diseases", "combo_to_points", "combo_detail"):
        if not combos:
            if diseases:
                query_type = "disease_to_plans"
            elif points:
                query_type = "point_to_combos"
            else:
                query_type = "unknown"

    elif query_type == "point_to_combos":
        if not points:
            if combos:
                query_type = "combo_to_points"
            elif diseases:
                query_type = "disease_to_plans"
            else:
                query_type = "unknown"

    elif query_type == "disease_combo_to_effect":
        if not combos and diseases:
            query_type = "disease_to_plans"
        elif not diseases and combos:
            query_type = "combo_to_diseases"
        elif not diseases and not combos:
            if points:
                query_type = "point_to_combos"
            else:
                query_type = "unknown"

    elif query_type == "multi_disease_common":
        if len(diseases) < 2:
            if len(diseases) == 1:
                query_type = "disease_to_plans"
            elif combos:
                query_type = "combo_to_diseases"
            else:
                query_type = "unknown"

    if query_type != original:
        logger.warning(
            f"query_type 自动降级: '{original}' → '{query_type}' "
            f"(diseases={diseases}, combos={combos}, points={points})"
        )

    return query_type


# ==========================
# Cypher 构造与执行
# ==========================
def run_cypher(cql: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    logger.info(f"Cypher: {cql}  params={params}")
    data = graph.run(cql, **params).data()
    logger.info(f"Records: {len(data)}")
    return data


def _build_disease_to_plans(
    disease_name: str,
    max_plans: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    cql_plans = """
    MATCH (d:Disease {name: $disease_name})-[:HAS_PLAN]->(p:TreatmentPlan)
    WITH d, p ORDER BY p.plan_id LIMIT $limit

    CALL {
        WITH p
        OPTIONAL MATCH (p)-[:MAIN_POINT]->(mc:AcupointCombo)
        RETURN collect(DISTINCT mc.name) AS main_combos
    }
    CALL {
        WITH p
        OPTIONAL MATCH (p)-[:AUX_POINT]->(ac:AcupointCombo)
        RETURN collect(DISTINCT ac.name) AS aux_combos
    }
    CALL {
        WITH p
        OPTIONAL MATCH (p)-[:MAIN_POINT]->(mp:Acupoint)
        RETURN collect(DISTINCT mp.name) AS main_points
    }
    CALL {
        WITH p
        OPTIONAL MATCH (p)-[:AUX_POINT]->(ap:Acupoint)
        RETURN collect(DISTINCT ap.name) AS aux_points
    }
    CALL {
        WITH p
        OPTIONAL MATCH (p)-[:MAIN_POINT]->(mc2:AcupointCombo)-[:HAS_POINT]->(msp:Acupoint)
        RETURN collect(DISTINCT msp.name) AS main_combo_std_points
    }
    CALL {
        WITH p
        OPTIONAL MATCH (p)-[:MAIN_POINT]->(mc3:AcupointCombo)-[:HAS_LOCAL_POINT]->(mlp:ComboLocalPoint)
        RETURN collect(DISTINCT mlp.name) AS main_combo_local_points
    }
    CALL {
        WITH p
        OPTIONAL MATCH (p)-[:MAIN_POINT]->(mc4:AcupointCombo)-[:HAS_COMBO]->(msc:AcupointCombo)
        RETURN collect(DISTINCT msc.name) AS main_combo_sub_combos
    }
    CALL {
        WITH p
        OPTIONAL MATCH (p)-[:AUX_POINT]->(ac2:AcupointCombo)-[:HAS_POINT]->(asp:Acupoint)
        RETURN collect(DISTINCT asp.name) AS aux_combo_std_points
    }
    CALL {
        WITH p
        OPTIONAL MATCH (p)-[:AUX_POINT]->(ac3:AcupointCombo)-[:HAS_LOCAL_POINT]->(alp:ComboLocalPoint)
        RETURN collect(DISTINCT alp.name) AS aux_combo_local_points
    }
    CALL {
        WITH p
        OPTIONAL MATCH (p)-[:AUX_POINT]->(ac4:AcupointCombo)-[:HAS_COMBO]->(asc:AcupointCombo)
        RETURN collect(DISTINCT asc.name) AS aux_combo_sub_combos
    }

    RETURN
        d.name AS disease,
        p.plan_id AS plan_id,
        p.method_text AS method_text,
        p.course_text AS course_text,
        p.effect_text AS effect_text,
        p.position_text AS position_text,
        p.effect_level AS effect_level,
        p.has_electroacupuncture AS has_electroacupuncture,
        p.has_moxibustion AS has_moxibustion,
        p.has_drug AS has_drug,
        main_combos, aux_combos, main_points, aux_points,
        main_combo_std_points, main_combo_local_points, main_combo_sub_combos,
        aux_combo_std_points, aux_combo_local_points, aux_combo_sub_combos
    """

    cql_summary = """
    MATCH (d:Disease {name: $disease_name})
    OPTIONAL MATCH (d)-[:MAIN_POINT_SUMMARY]->(mp)
    OPTIONAL MATCH (d)-[:AUX_POINT_SUMMARY]->(ap)
    RETURN
        collect(DISTINCT CASE WHEN mp:Acupoint      THEN mp.name END) AS disease_main_points,
        collect(DISTINCT CASE WHEN mp:AcupointCombo THEN mp.name END) AS disease_main_combos,
        collect(DISTINCT CASE WHEN ap:Acupoint      THEN ap.name END) AS disease_aux_points,
        collect(DISTINCT CASE WHEN ap:AcupointCombo THEN ap.name END) AS disease_aux_combos
    """

    params = {"disease_name": disease_name, "limit": max_plans}
    plan_records = run_cypher(cql_plans, params)
    summary_records = run_cypher(cql_summary, {"disease_name": disease_name})

    summary = summary_records[0] if summary_records else {}
    for r in plan_records:
        r.update(summary)

    return cql_plans, plan_records


def build_and_run_query(
    parsed: Dict[str, Any],
    max_plans: int = 10,
) -> Tuple[str, List[Dict[str, Any]]]:
    qt = parsed["query_type"]
    diseases = parsed.get("diseases") or []
    combos = parsed.get("combos") or []
    points = parsed.get("points") or []

    if qt == "disease_to_plans":
        if not diseases:
            logger.warning("disease_to_plans 缺少疾病实体，返回空结果")
            return "", []
        all_records: List[Dict[str, Any]] = []
        cql_used = ""
        for disease_name in diseases:
            cql, recs = _build_disease_to_plans(disease_name, max_plans)
            cql_used = cql
            all_records.extend(recs)
        return cql_used, all_records

    elif qt == "combo_to_diseases":
        if not combos:
            logger.warning("combo_to_diseases 缺少组合实体，返回空结果")
            return "", []
        cql = """
        MATCH (c:AcupointCombo {name: $combo_name})
        OPTIONAL MATCH (d:Disease)-[:HAS_PLAN]->(p:TreatmentPlan)
         WHERE (p)-[:MAIN_POINT]->(c) OR (p)-[:AUX_POINT]->(c)
        RETURN
            c.name AS combo_name,
            c.indications AS combo_indications,
            c.acupuncture_method_json AS combo_acu_method,
            collect(DISTINCT d.name) AS diseases,
            collect(DISTINCT p.plan_id) AS plan_ids
        """
        all_records = []
        for combo_name in combos:
            all_records.extend(run_cypher(cql, {"combo_name": combo_name}))
        return cql, all_records

    elif qt == "disease_combo_to_effect":
        if not diseases or not combos:
            logger.warning("disease_combo_to_effect 缺少必要实体，返回空结果")
            return "", []
        cql = """
        MATCH (d:Disease {name: $disease_name})-[:HAS_PLAN]->(p:TreatmentPlan)
        MATCH (c:AcupointCombo {name: $combo_name})
        WHERE (p)-[:MAIN_POINT]->(c) OR (p)-[:AUX_POINT]->(c)
        RETURN
            d.name AS disease, c.name AS combo_name,
            p.plan_id AS plan_id, p.method_text AS method_text,
            p.course_text AS course_text, p.effect_text AS effect_text,
            p.position_text AS position_text, p.effect_level AS effect_level,
            p.has_electroacupuncture AS has_electroacupuncture,
            p.has_moxibustion AS has_moxibustion, p.has_drug AS has_drug
        LIMIT $limit
        """
        all_records = []
        for d in diseases:
            for c in combos:
                all_records.extend(run_cypher(cql, {"disease_name": d, "combo_name": c, "limit": max_plans}))
        return cql, all_records

    elif qt == "point_to_combos":
        if not points:
            logger.warning("point_to_combos 缺少穴位实体，返回空结果")
            return "", []
        cql = """
        MATCH (a:Acupoint {name: $point_name})
        OPTIONAL MATCH (combo:AcupointCombo)-[:HAS_POINT]->(a)
        OPTIONAL MATCH (p_main:TreatmentPlan)-[:MAIN_POINT]->(a)
        OPTIONAL MATCH (d_main:Disease)-[:HAS_PLAN]->(p_main)
        OPTIONAL MATCH (p_aux:TreatmentPlan)-[:AUX_POINT]->(a)
        OPTIONAL MATCH (d_aux:Disease)-[:HAS_PLAN]->(p_aux)
        RETURN
            a.name AS point_name,
            collect(DISTINCT combo.name)  AS combos,
            collect(DISTINCT d_main.name) AS diseases_as_main,
            collect(DISTINCT d_aux.name)  AS diseases_as_aux,
            count(DISTINCT p_main)        AS main_plan_count,
            count(DISTINCT p_aux)         AS aux_plan_count
        """
        all_records = []
        for point_name in points:
            all_records.extend(run_cypher(cql, {"point_name": point_name}))
        return cql, all_records

    elif qt == "disease_to_point_summary":
        if not diseases:
            logger.warning("disease_to_point_summary 缺少疾病实体，返回空结果")
            return "", []
        cql = """
        MATCH (d:Disease {name: $disease_name})
        OPTIONAL MATCH (d)-[:MAIN_POINT_SUMMARY]->(mp)
        OPTIONAL MATCH (d)-[:AUX_POINT_SUMMARY]->(ap)
        RETURN
            d.name AS disease,
            collect(DISTINCT CASE WHEN mp:Acupoint      THEN mp.name END) AS main_points,
            collect(DISTINCT CASE WHEN mp:AcupointCombo THEN mp.name END) AS main_combos,
            collect(DISTINCT CASE WHEN ap:Acupoint      THEN ap.name END) AS aux_points,
            collect(DISTINCT CASE WHEN ap:AcupointCombo THEN ap.name END) AS aux_combos
        """
        all_records = []
        for d in diseases:
            all_records.extend(run_cypher(cql, {"disease_name": d}))
        return cql, all_records

    elif qt == "combo_to_points":
        if not combos:
            logger.warning("combo_to_points 缺少组合实体，返回空结果")
            return "", []
        cql = """
        MATCH (c:AcupointCombo {name: $combo_name})
        CALL {
            WITH c
            OPTIONAL MATCH (c)-[r1:HAS_POINT]->(stdPoint:Acupoint)
            RETURN collect(DISTINCT {name: stdPoint.name, needle_method: r1.needle_method}) AS std_points
        }
        CALL {
            WITH c
            OPTIONAL MATCH (c)-[r2:HAS_LOCAL_POINT]->(localPoint:ComboLocalPoint)
            RETURN collect(DISTINCT {name: localPoint.name, needle_method: r2.needle_method}) AS local_points
        }
        CALL {
            WITH c
            OPTIONAL MATCH (c)-[:HAS_COMBO]->(subCombo:AcupointCombo)
            RETURN collect(DISTINCT subCombo.name) AS sub_combos
        }
        RETURN
            c.name AS combo_name, c.indications AS indications,
            c.acupuncture_method_json AS acupuncture_method,
            std_points, local_points, sub_combos
        """
        all_records = []
        for c in combos:
            all_records.extend(run_cypher(cql, {"combo_name": c}))
        return cql, all_records

    elif qt == "disease_compare_plans":
        if not diseases:
            logger.warning("disease_compare_plans 缺少疾病实体，返回空结果")
            return "", []
        cql = """
        MATCH (d:Disease {name: $disease_name})-[:HAS_PLAN]->(p:TreatmentPlan)
        CALL {
            WITH p
            OPTIONAL MATCH (p)-[:MAIN_POINT]->(mc:AcupointCombo)
            RETURN collect(DISTINCT mc.name) AS main_combos
        }
        CALL {
            WITH p
            OPTIONAL MATCH (p)-[:AUX_POINT]->(ac:AcupointCombo)
            RETURN collect(DISTINCT ac.name) AS aux_combos
        }
        CALL {
            WITH p
            OPTIONAL MATCH (p)-[:MAIN_POINT]->(mp:Acupoint)
            RETURN collect(DISTINCT mp.name) AS main_points
        }
        CALL {
            WITH p
            OPTIONAL MATCH (p)-[:AUX_POINT]->(ap:Acupoint)
            RETURN collect(DISTINCT ap.name) AS aux_points
        }
        RETURN
            d.name AS disease, p.plan_id AS plan_id,
            p.method_text AS method_text, p.course_text AS course_text,
            p.effect_text AS effect_text, p.effect_level AS effect_level,
            p.has_electroacupuncture AS has_electroacupuncture,
            p.has_moxibustion AS has_moxibustion, p.has_drug AS has_drug,
            main_combos, aux_combos, main_points, aux_points
        ORDER BY p.effect_level DESC, p.plan_id
        LIMIT $limit
        """
        all_records = []
        for d in diseases:
            all_records.extend(run_cypher(cql, {"disease_name": d, "limit": max_plans}))
        return cql, all_records

    elif qt == "combo_detail":
        if not combos:
            logger.warning("combo_detail 缺少组合实体，返回空结果")
            return "", []
        cql = """
        MATCH (c:AcupointCombo {name: $combo_name})
        CALL {
            WITH c
            OPTIONAL MATCH (c)-[r1:HAS_POINT]->(stdPoint:Acupoint)
            RETURN collect(DISTINCT {name: stdPoint.name, needle_method: r1.needle_method}) AS std_points
        }
        CALL {
            WITH c
            OPTIONAL MATCH (c)-[r2:HAS_LOCAL_POINT]->(localPoint:ComboLocalPoint)
            RETURN collect(DISTINCT {name: localPoint.name, needle_method: r2.needle_method}) AS local_points
        }
        CALL {
            WITH c
            OPTIONAL MATCH (c)-[:HAS_COMBO]->(subCombo:AcupointCombo)
            RETURN collect(DISTINCT subCombo.name) AS sub_combos
        }
        CALL {
            WITH c
            OPTIONAL MATCH (d:Disease)-[:HAS_PLAN]->(p:TreatmentPlan)
             WHERE (p)-[:MAIN_POINT]->(c) OR (p)-[:AUX_POINT]->(c)
            RETURN collect(DISTINCT d.name) AS related_diseases, count(DISTINCT p) AS plan_count
        }
        RETURN
            c.name AS combo_name, c.indications AS indications,
            c.acupuncture_method_json AS acupuncture_method,
            std_points, local_points, sub_combos,
            related_diseases, plan_count
        """
        all_records = []
        for c in combos:
            all_records.extend(run_cypher(cql, {"combo_name": c}))
        return cql, all_records

    elif qt == "multi_disease_common":
        if len(diseases) < 2:
            logger.warning("multi_disease_common 不足两个疾病，返回空结果")
            return "", []
        cql_simple = """
        MATCH (d:Disease {name: $disease_name})-[:HAS_PLAN]->(p:TreatmentPlan)
        OPTIONAL MATCH (p)-[:MAIN_POINT]->(mc:AcupointCombo)
        OPTIONAL MATCH (p)-[:AUX_POINT]->(ac:AcupointCombo)
        OPTIONAL MATCH (p)-[:MAIN_POINT]->(mp:Acupoint)
        OPTIONAL MATCH (p)-[:AUX_POINT]->(ap:Acupoint)
        WITH d.name AS disease,
             collect(DISTINCT mc.name) + collect(DISTINCT ac.name) AS all_combos,
             collect(DISTINCT mp.name) + collect(DISTINCT ap.name) AS all_points
        RETURN disease,
               [x IN all_combos WHERE x IS NOT NULL] AS all_combos,
               [x IN all_points WHERE x IS NOT NULL] AS all_points
        """
        per_disease = []
        for d in diseases:
            recs = run_cypher(cql_simple, {"disease_name": d})
            if recs:
                per_disease.append(recs[0])

        if len(per_disease) >= 2:
            combo_sets = [set(r.get("all_combos") or []) - {None} for r in per_disease]
            point_sets = [set(r.get("all_points") or []) - {None} for r in per_disease]
            common_combos = sorted(combo_sets[0].intersection(*combo_sets[1:]))
            common_points = sorted(point_sets[0].intersection(*point_sets[1:]))
            result = {
                "diseases": diseases,
                "common_combos": common_combos,
                "common_points": common_points,
                "per_disease_detail": per_disease,
            }
            return cql_simple, [result]
        else:
            return cql_simple, per_disease

    else:
        logger.warning(f"未知 query_type: {qt}")
        return "", []


# ==========================
# 空结果回退
# ==========================
def _fallback_search(
    parsed: Dict[str, Any],
    max_plans: int = 10,
) -> Tuple[str, List[Dict[str, Any]]]:
    qt = parsed["query_type"]
    diseases = parsed.get("diseases") or []
    combos = parsed.get("combos") or []
    points = parsed.get("points") or []

    if qt in ("disease_to_plans", "disease_to_point_summary",
              "disease_combo_to_effect", "disease_compare_plans") and diseases:
        keyword = diseases[0]
        for kw in [keyword, keyword[:3], keyword[:2]]:
            if not kw:
                continue
            fallback_cql = """
            MATCH (d:Disease)-[:HAS_PLAN]->(p:TreatmentPlan)
            WHERE d.name CONTAINS $keyword
            OPTIONAL MATCH (p)-[:MAIN_POINT]->(mc:AcupointCombo)
            RETURN d.name AS disease, p.plan_id AS plan_id,
                   p.effect_text AS effect_text, p.effect_level AS effect_level,
                   collect(DISTINCT mc.name) AS main_combos
            ORDER BY p.plan_id LIMIT $limit
            """
            records = run_cypher(fallback_cql, {"keyword": kw, "limit": max_plans})
            if records:
                logger.info(f"Fallback 命中（disease）：keyword='{kw}', {len(records)} records")
                return fallback_cql, records

    if qt in ("combo_to_diseases", "combo_to_points", "combo_detail") and combos:
        keyword = combos[0]
        for kw in [keyword, keyword[:3], keyword[:2]]:
            if not kw:
                continue
            fallback_cql = """
            MATCH (c:AcupointCombo) WHERE c.name CONTAINS $keyword
            OPTIONAL MATCH (d:Disease)-[:HAS_PLAN]->(p:TreatmentPlan)
             WHERE (p)-[:MAIN_POINT]->(c) OR (p)-[:AUX_POINT]->(c)
            RETURN c.name AS combo_name, c.indications AS indications,
                   collect(DISTINCT d.name) AS diseases
            LIMIT $limit
            """
            records = run_cypher(fallback_cql, {"keyword": kw, "limit": max_plans})
            if records:
                logger.info(f"Fallback 命中（combo）：keyword='{kw}', {len(records)} records")
                return fallback_cql, records

    if qt == "point_to_combos" and points:
        keyword = points[0]
        fallback_cql = """
        MATCH (a:Acupoint) WHERE a.name CONTAINS $keyword
        OPTIONAL MATCH (combo:AcupointCombo)-[:HAS_POINT]->(a)
        RETURN a.name AS point_name, collect(DISTINCT combo.name) AS combos
        LIMIT $limit
        """
        records = run_cypher(fallback_cql, {"keyword": keyword, "limit": max_plans})
        if records:
            logger.info(f"Fallback 命中（point）：keyword='{keyword}', {len(records)} records")
            return fallback_cql, records

    return "", []


# ==========================
# FastAPI 应用
# ==========================
app = FastAPI(
    title="Jin's Three-Needle KG+LLM QA System",
    description=(
        "基于靳三针知识图谱与 Qwen3 的智能问答后端"
        "（v2.3 分层可信度生成 + LLM 实体对齐 + 多轮对话 + 降级版）。"
    ),
    version="2.3.0",
)

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/qa", response_model=QAResponse)
def qa_endpoint(req: QARequest):
    try:
        question = (req.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空。")

        history_for_llm: Optional[List[Dict[str, str]]] = None
        if req.history:
            history_for_llm = [
                {"role": h.role, "content": h.content}
                for h in req.history
                if h.role in ("user", "assistant")
            ]

        # 1. LLM 解析（带 history 做指代消解）
        parsed = call_llm_for_parse(question, history_for_llm)

        # 2. 实体对齐（精确 → LLM → difflib）
        parsed_norm, entity_suggest = normalize_entities(parsed)
        query_type = parsed_norm["query_type"]
        entities = {
            "diseases": parsed_norm.get("diseases") or [],
            "combos": parsed_norm.get("combos") or [],
            "points": parsed_norm.get("points") or [],
        }

        # ★ 3. 自动降级不兼容的 query_type
        query_type = _adjust_query_type(query_type, entities)
        parsed_norm["query_type"] = query_type

        use_kg = True if req.use_kg is None else bool(req.use_kg)

        # ========== 分支 1：不使用 KG（纯 LLM） ==========
        if not use_kg:
            ans = call_llm_for_answer_no_kg(
                question=question, query_type=query_type,
                entities=entities, history=history_for_llm,
            )
            return QAResponse(
                answer=ans, query_type=query_type, entities=entities,
                entity_suggest=entity_suggest or None,
                answer_source="llm_only",
            )

        # ========== 分支 2：使用 KG（分层可信度生成） ==========
        if query_type == "unknown":
            ans = call_llm_for_answer(
                question=question, records=[], query_type=query_type,
                entities=entities, history=history_for_llm,
            )
            return QAResponse(
                answer=ans, query_type=query_type, entities=entities,
                records=[], entity_suggest=entity_suggest or None,
                answer_source="kg_hybrid",
            )

        # 4. Cypher 查询
        max_plans = req.options.max_plans if req.options else 10
        cypher, records = build_and_run_query(parsed_norm, max_plans=max_plans)

        # 5. 空结果回退
        is_fallback = False
        if not records:
            logger.info("主查询无结果，尝试 fallback...")
            fallback_cypher, fallback_records = _fallback_search(parsed_norm, max_plans=max_plans)
            if fallback_records:
                cypher = fallback_cypher
                records = fallback_records
                is_fallback = True

        # 6. 清洗 + 分层可信度生成
        cleaned_records = _clean_records(records)
        ans = call_llm_for_answer(
            question=question, records=cleaned_records,
            query_type=query_type, entities=entities, history=history_for_llm,
        )

        if is_fallback:
            ans = "【提示】未找到精确匹配的知识图谱记录，以下结果基于模糊检索，仅供参考。\n\n" + ans

        return QAResponse(
            answer=ans, query_type=query_type, entities=entities,
            cypher=cypher or None, records=records,
            entity_suggest=entity_suggest or None,
            answer_source="kg_hybrid",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("QA 处理发生错误")
        raise HTTPException(status_code=500, detail=f"服务器内部错误：{e}")


@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 靳三针知识图谱问答系统后端启动中...")
    print("=" * 60)
    print(f"📌 API 文档地址：http://localhost:8000/docs")
    print(f"📌 健康检查：http://localhost:8000/health")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
