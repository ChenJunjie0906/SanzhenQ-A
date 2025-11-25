from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import os
import json
import logging
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
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
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
    role: str     # user / assistant
    content: str


class QAOptions(BaseModel):
    max_plans: Optional[int] = 10
    language: Optional[str] = "zh"


class QARequest(BaseModel):
    question: str
    history: Optional[List[HistoryItem]] = None
    options: Optional[QAOptions] = None
    use_kg: Optional[bool] = True  # 是否使用知识图谱，默认 True

class QAResponse(BaseModel):
    answer: str
    query_type: Optional[str] = None
    entities: Optional[Dict[str, List[str]]] = None
    cypher: Optional[str] = None
    records: Optional[List[Dict[str, Any]]] = None
    entity_suggest: Optional[Dict[str, Any]] = None


# ==========================
# 实体缓存 & 模糊对齐
# ==========================
@lru_cache()
def get_all_diseases() -> List[str]:
    cql = "MATCH (d:Disease) RETURN DISTINCT d.name AS name"
    data = graph.run(cql).data()
    return [r["name"] for r in data if r.get("name")]


@lru_cache()
def get_all_combos() -> List[str]:
    cql = "MATCH (c:AcupointCombo) RETURN DISTINCT c.name AS name"
    data = graph.run(cql).data()
    return [r["name"] for r in data if r.get("name")]


@lru_cache()
def get_all_points() -> List[str]:
    cql = "MATCH (a:Acupoint) RETURN DISTINCT a.name AS name"
    data = graph.run(cql).data()
    return [r["name"] for r in data if r.get("name")]


def _best_match(
    name: str,
    candidates: List[str],
    cutoff: float = 0.7,
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


def normalize_entities(
    parsed: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    将 LLM 解析的实体（文本）对齐到图谱中的 Disease / AcupointCombo / Acupoint 节点 name。
    """
    normalized = parsed.copy()
    suggest: Dict[str, Any] = {
        "unmatched": [],
        "candidates": {},
    }

    # 疾病
    all_d = get_all_diseases()
    norm_d = []
    for d in parsed.get("diseases") or []:
        best, cands = _best_match(d, all_d, cutoff=0.7)
        if best:
            norm_d.append(best)
            if len(cands) > 1:
                suggest["candidates"].setdefault(d, {})["diseases"] = cands
            if best != d:
                logger.info(f"Disease 对齐: '{d}' -> '{best}'")
        else:
            norm_d.append(d)
            suggest["unmatched"].append({"type": "disease", "name": d})
            if cands:
                suggest["candidates"].setdefault(d, {})["diseases"] = cands
    normalized["diseases"] = norm_d

    # 组合
    all_c = get_all_combos()
    norm_c = []
    for c in parsed.get("combos") or []:
        best, cands = _best_match(c, all_c, cutoff=0.7)
        if best:
            norm_c.append(best)
            if len(cands) > 1:
                suggest["candidates"].setdefault(c, {})["combos"] = cands
            if best != c:
                logger.info(f"Combo 对齐: '{c}' -> '{best}'")
        else:
            norm_c.append(c)
            suggest["unmatched"].append({"type": "combo", "name": c})
            if cands:
                suggest["candidates"].setdefault(c, {})["combos"] = cands
    normalized["combos"] = norm_c

    # 穴位（Acupoint）
    all_p = get_all_points()
    norm_p = []
    for p in parsed.get("points") or []:
        best, cands = _best_match(p, all_p, cutoff=0.7)
        if best:
            norm_p.append(best)
            if len(cands) > 1:
                suggest["candidates"].setdefault(p, {})["points"] = cands
            if best != p:
                logger.info(f"Acupoint 对齐: '{p}' -> '{best}'")
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
    """
    不依赖图谱 records 的版本：只用 LLM + 问题 + 解析出的结构来回答。
    用于和 KG+LLM 做对比评测。
    """
    if query_type == "unknown":
        return (
            "当前无法从问题中提取出明确的疾病、靳三针组合或穴位信息，"
            "建议你补充更具体的描述，例如“脑卒中后遗症常用哪些靳三针方案？”。"
        )

    system_prompt = """
你是一个靳三针针灸领域的学术型问答助手。
当前模式为“仅 LLM（不依赖知识图谱）”，
请基于你已有的通用医学知识和中医针灸知识，
围绕用户问题进行回答，重点说明靳三针相关内容。

要求：
- 用中文专业、客观地回答，尽量结构清晰（可分点、分条）。
- 可以结合你已知的靳三针理论、常见取穴组合、常见适应证和疗效结论进行总结。
- 不能声称“来自本系统图谱”，也不要编造具体文献标题或编号。
- 不要给出具体处方或个体化诊疗建议（如“你应该去扎哪里”这类直指个人的说法）。
- 回答最后同样加一句提醒：
  “以上内容主要基于通用中医针灸与靳三针相关知识，供教学与科研参考，不构成个体化诊疗建议。”
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
            r = h.get("role")
            c = h.get("content", "")
            if r in ("user", "assistant"):
                messages.append({"role": r, "content": c})

    messages.append(
        {
            "role": "user",
            "content": "下面是本次查询的结构化解析数据，请结合这些信息并依靠你自身知识回答：\n"
                       + json.dumps(user_payload, ensure_ascii=False, indent=2),
        }
    )

    try:
        ans = _chat_completion(
            model=QWEN_MODEL_ANSWER,
            messages=messages,
            temperature=0.5,
        )
        return ans
    except Exception:
        logger.exception("LLM 生成答案失败（no KG 模式），使用降级文本")
        return (
            "当前在未调用知识图谱的前提下生成答案失败，"
            "建议稍后重试或在有 KG 模式下查看回答。"
        )
def call_llm_for_parse(
    question: str,
    history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    问题解析：输出 query_type / diseases / combos / points
    """
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
- "disease_to_plans"          ：问“某个疾病有哪些靳三针治疗方案”
- "combo_to_diseases"         ：问“某个靳三针组合能治什么病”
- "disease_combo_to_effect"   ：问“某个组合治疗某病的疗效如何”
- "point_to_combos"           ：问“某个穴位参与哪些靳三针组合/用于哪些病”
- "disease_to_point_summary"  ：问“某病靳三针常用主穴/配穴总结”
- "combo_to_points"           ：问“某个靳三针组合包含哪些穴位/子组合”
说明：
- diseases：提取文本中的疾病名称，如“脑卒中后遗症”、“失眠”等。
- combos：提取靳三针“组合名称”，如“颞三针”、“智三针”、“自闭九项”等。
- points：提取单个标准穴位名，如“合谷”、“足三里”，不要把“三针组合”算作点。

必须：
1）只返回 JSON，本身不加任何注释文字；
2）所有字段都要出现，即使为空也要给空数组。
"""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt.strip()}
    ]
    if history:
        for h in history:
            r = h.get("role")
            c = h.get("content", "")
            if r in ("user", "assistant"):
                messages.append({"role": r, "content": c})

    messages.append({"role": "user", "content": question})

    try:
        content = _chat_completion(
            model=QWEN_MODEL_PARSE,
            messages=messages,
            temperature=0.1,
        )
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

        # 非常简单的兜底策略（只处理少数典型情况）
        if "颞三针" in q and ("什么病" in q or "哪些病" in q or "适应证" in q):
            return {
                "query_type": "combo_to_diseases",
                "diseases": [],
                "combos": ["颞三针"],
                "points": [],
            }
        if "脑卒中" in q:
            return {
                "query_type": "disease_to_plans",
                "diseases": ["脑卒中后遗症"],
                "combos": [],
                "points": [],
            }
        return {
            "query_type": "unknown",
            "diseases": [],
            "combos": [],
            "points": [],
        }


def call_llm_for_answer(
    question: str,
    records: List[Dict[str, Any]],
    query_type: str,
    entities: Dict[str, List[str]],
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    if query_type == "unknown":
        return (
            "当前无法从问题中提取出明确的疾病、靳三针组合或穴位信息，"
            "建议你补充更具体的描述，例如“脑卒中后遗症常用哪些靳三针方案？”。"
        )

    if not records:
        return (
            "在目前已构建的靳三针知识图谱中，暂未检索到与该问题直接对应的结构化记录，"
            "可能是相关文献尚未收录或仍在整理中。"
            "请结合权威教材和临床指南进一步查证，本系统仅用于科研与教学参考。"
        )

    system_prompt = """
你是一个靳三针知识图谱问答助手。现在提供给你：
1）用户原始问题（当前轮）；
2）必要时提供的前文对话历史（多轮上下文）；
3）从 Neo4j 知识图谱检索到的 records（JSON 列表）；
4）解析出的 query_type 和实体（疾病、组合、穴位）。

请严格根据 records 中的信息，用中文生成一个“学术、客观、不过度推断、不给诊断/处方”的回答。
不要杜撰 records 里没有的结论。

不同 query_type 时的侧重点：
- disease_to_plans：
    围绕“该疾病有哪些靳三针治疗方案”，概括主穴组合、辅助穴位、电针/艾灸/药物配合、疗程和疗效结论等。
- combo_to_diseases：
    总结“该靳三针组合主要用于哪些疾病或病证”，可按系统或疾病列举。
- disease_combo_to_effect：
    围绕“该组合治疗该病的疗效证据”，说明研究设计（若有）、疗程、主要疗效指标。
- point_to_combos：
    总结“该穴位参与了哪些靳三针组合，这些组合主要用于哪些疾病”。
- disease_to_point_summary：
    以提纲式列出某病常用主穴、配穴（可区分主穴/配穴），不需要详细操作方法。
- combo_to_points：
    总结“该靳三针组合由哪些标准穴位、局部点和子组合构成”，
    可按主穴、局部点、子组合分类列出。

必须：
- 严格基于 records 中的字段，例如：
  - disease, plan_id, method_text, course_text, effect_text, effect_level,
    has_electroacupuncture, has_moxibustion, has_drug,
    main_combos, aux_combos, main_points, aux_points,
    combos, diseases 等。
- 可以适当合并相似方案做总结，但不要推断不存在的适应证。

回答最后加一句提醒：
“以上内容仅基于已纳入的靳三针知识图谱与相关文献，供教学与科研参考，不构成个体化诊疗建议。”
"""
    user_payload = {
        "question": question,
        "query_type": query_type,
        "entities": entities,
        "records": records,
    }

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt.strip()},
    ]

    # 新增：如果有多轮历史，先喂给模型，帮助理解“那”、“这个”等指代
    if history:
        for h in history:
            r = h.get("role")
            c = h.get("content", "")
            if r in ("user", "assistant"):
                messages.append({"role": r, "content": c})

    # 当前轮问题 + 本轮的结构化数据
    messages.append(
        {
            "role": "user",
            "content": "下面是本次查询的完整数据，请在理解以上对话历史的基础上作答：\n"
                       + json.dumps(user_payload, ensure_ascii=False, indent=2),
        }
    )

    try:
        ans = _chat_completion(
            model=QWEN_MODEL_ANSWER,
            messages=messages,
            temperature=0.3,
        )
        return ans
    except Exception:
        logger.exception("LLM 生成答案失败，使用降级文本")
        return (
            "系统已成功从知识图谱中检索到若干条与问题相关的靳三针方案，"
            "但在生成自然语言总结时发生错误。"
            "建议直接查看原始 records 数据，并结合原文文献进行分析。"
        )


# ==========================
# Cypher 构造与执行（完全基于文中实体/关系）
# ==========================
def run_cypher(cql: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    logger.info(f"Cypher: {cql}  params={params}")
    data = graph.run(cql, **params).data()
    logger.info(f"Records: {len(data)}")
    return data


def build_and_run_query(
    parsed: Dict[str, Any],
    max_plans: int = 10
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    按设计的 6 类 query_type，使用 Disease / TreatmentPlan / Acupoint /
    AcupointCombo / ComboLocalPoint + 关系类型构造 Cypher。
    """
    qt = parsed["query_type"]
    diseases = parsed.get("diseases") or []
    combos = parsed.get("combos") or []
    points = parsed.get("points") or []

    # 1. 疾病 → 方案（Disease --HAS_PLAN--> TreatmentPlan）
    if qt == "disease_to_plans":
        if not diseases:
            raise ValueError("disease_to_plans 需要至少一个疾病实体。")
        disease_name = diseases[0]
        cql = """
                MATCH (d:Disease {name: $disease_name})-[:HAS_PLAN]->(p:TreatmentPlan)
                // 方案直接挂的组合 / 穴位
                OPTIONAL MATCH (p)-[:MAIN_POINT]->(mainCombo:AcupointCombo)
                OPTIONAL MATCH (p)-[:AUX_POINT]->(auxCombo:AcupointCombo)
                OPTIONAL MATCH (p)-[:MAIN_POINT]->(mainPoint:Acupoint)
                OPTIONAL MATCH (p)-[:AUX_POINT]->(auxPoint:Acupoint)

                // ========= 组合内部包含的穴位 / 子组合 =========
                OPTIONAL MATCH (mainCombo)-[:HAS_POINT]->(mainStdPoint:Acupoint)
                OPTIONAL MATCH (mainCombo)-[:HAS_LOCAL_POINT]->(mainLocalPoint:ComboLocalPoint)
                OPTIONAL MATCH (mainCombo)-[:HAS_COMBO]->(mainSubCombo:AcupointCombo)

                OPTIONAL MATCH (auxCombo)-[:HAS_POINT]->(auxStdPoint:Acupoint)
                OPTIONAL MATCH (auxCombo)-[:HAS_LOCAL_POINT]->(auxLocalPoint:ComboLocalPoint)
                OPTIONAL MATCH (auxCombo)-[:HAS_COMBO]->(auxSubCombo:AcupointCombo)
                // ===================================================

                // ========= 疾病级别的主穴/配穴汇总（*_SUMMARY） =========
                OPTIONAL MATCH (d)-[:MAIN_POINT_SUMMARY]->(mp_sum)
                OPTIONAL MATCH (d)-[:AUX_POINT_SUMMARY]->(ap_sum)
                // ===================================================

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

                    // 方案级：主/辅三针组合 & 穴位
                    collect(DISTINCT mainCombo.name) AS main_combos,
                    collect(DISTINCT auxCombo.name)  AS aux_combos,
                    collect(DISTINCT mainPoint.name) AS main_points,
                    collect(DISTINCT auxPoint.name)  AS aux_points,

                    // 方案级：主/辅组合内部包含的标准穴位、局部点、子组合
                    collect(DISTINCT mainStdPoint.name)        AS main_combo_std_points,
                    collect(DISTINCT mainLocalPoint.name)      AS main_combo_local_points,
                    collect(DISTINCT mainSubCombo.name)        AS main_combo_sub_combos,

                    collect(DISTINCT auxStdPoint.name)         AS aux_combo_std_points,
                    collect(DISTINCT auxLocalPoint.name)       AS aux_combo_local_points,
                    collect(DISTINCT auxSubCombo.name)         AS aux_combo_sub_combos,

                    // 疾病级：汇总主/辅穴位 & 主/辅组合（来自 *_SUMMARY）
                    collect(DISTINCT CASE WHEN mp_sum:Acupoint      THEN mp_sum.name END) AS disease_main_points,
                    collect(DISTINCT CASE WHEN mp_sum:AcupointCombo THEN mp_sum.name END) AS disease_main_combos,
                    collect(DISTINCT CASE WHEN ap_sum:Acupoint      THEN ap_sum.name END) AS disease_aux_points,
                    collect(DISTINCT CASE WHEN ap_sum:AcupointCombo THEN ap_sum.name END) AS disease_aux_combos

                ORDER BY p.plan_id
                LIMIT $limit
                """
        params = {"disease_name": disease_name, "limit": max_plans}
        return cql, run_cypher(cql, params)

    # 2. 组合 → 疾病（AcupointCombo <-MAIN_POINT/AUX_POINT- TreatmentPlan <-HAS_PLAN- Disease）
    elif qt == "combo_to_diseases":
        if not combos:
            raise ValueError("combo_to_diseases 需要至少一个靳三针组合实体。")
        combo_name = combos[0]
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
        params = {"combo_name": combo_name}
        return cql, run_cypher(cql, params)

    # 3. 疾病 + 组合 → 疗效（Disease -HAS_PLAN- TreatmentPlan -MAIN/AUX-> AcupointCombo）
    elif qt == "disease_combo_to_effect":
        if not diseases or not combos:
            raise ValueError("disease_combo_to_effect 需要同时包含疾病和组合实体。")
        disease_name = diseases[0]
        combo_name = combos[0]
        cql = """
        MATCH (d:Disease {name: $disease_name})-[:HAS_PLAN]->(p:TreatmentPlan)
        MATCH (c:AcupointCombo {name: $combo_name})
        WHERE (p)-[:MAIN_POINT]->(c) OR (p)-[:AUX_POINT]->(c)
        RETURN
            d.name AS disease,
            c.name AS combo_name,
            p.plan_id AS plan_id,
            p.method_text AS method_text,
            p.course_text AS course_text,
            p.effect_text AS effect_text,
            p.position_text AS position_text,
            p.effect_level AS effect_level,
            p.has_electroacupuncture AS has_electroacupuncture,
            p.has_moxibustion AS has_moxibustion,
            p.has_drug AS has_drug
        LIMIT $limit
        """
        params = {
            "disease_name": disease_name,
            "combo_name": combo_name,
            "limit": max_plans,
        }
        return cql, run_cypher(cql, params)

    # 4. 穴位 / 局部点 → 组合 / 疾病
    #    (Acupoint|ComboLocalPoint) <-HAS_POINT/HAS_LOCAL_POINT- AcupointCombo
    #     <-MAIN/AUX_POINT- TreatmentPlan <-HAS_PLAN- Disease
    elif qt == "combo_to_points":
        if not combos:
            raise ValueError("combo_to_points 需要至少一个靳三针组合实体。")
        combo_name = combos[0]
        cql = """
                MATCH (c:AcupointCombo {name: $combo_name})
                OPTIONAL MATCH (c)-[r1:HAS_POINT]->(stdPoint:Acupoint)
                OPTIONAL MATCH (c)-[r2:HAS_LOCAL_POINT]->(localPoint:ComboLocalPoint)
                OPTIONAL MATCH (c)-[:HAS_COMBO]->(subCombo:AcupointCombo)
                RETURN
                    c.name AS combo_name,
                    // 标准腧穴 + 刺法说明
                    collect(
                        DISTINCT {
                            name: stdPoint.name,
                            needle_method: r1.needle_method
                        }
                    ) AS std_points,
                    // 局部点 + 刺法说明
                    collect(
                        DISTINCT {
                            name: localPoint.name,
                            needle_method: r2.needle_method
                        }
                    ) AS local_points,
                    // 子组合名称
                    collect(DISTINCT subCombo.name) AS sub_combos
                """
        params = {"combo_name": combo_name}
        return cql, run_cypher(cql, params)

    # 5. 疾病 → 主穴/配穴汇总（Disease -MAIN/AUX_POINT_SUMMARY-> Acupoint / AcupointCombo）
    elif qt == "disease_to_point_summary":
        if not diseases:
            raise ValueError("disease_to_point_summary 需要至少一个疾病实体。")
        disease_name = diseases[0]
        cql = """
        MATCH (d:Disease {name: $disease_name})
        OPTIONAL MATCH (d)-[:MAIN_POINT_SUMMARY]->(mp)
        OPTIONAL MATCH (d)-[:AUX_POINT_SUMMARY]->(ap)
        RETURN
            d.name AS disease,
            // 主穴：Acupoint
            collect(DISTINCT CASE WHEN mp:Acupoint      THEN mp.name END) AS main_points,
            // 主组合：AcupointCombo
            collect(DISTINCT CASE WHEN mp:AcupointCombo THEN mp.name END) AS main_combos,
            // 辅助穴位
            collect(DISTINCT CASE WHEN ap:Acupoint      THEN ap.name END) AS aux_points,
            // 辅助组合
            collect(DISTINCT CASE WHEN ap:AcupointCombo THEN ap.name END) AS aux_combos
        """
        params = {"disease_name": disease_name}
        return cql, run_cypher(cql, params)

    # 6. 组合 → 内部穴位 / 子组合
    elif qt == "combo_to_points":
        if not combos:
            raise ValueError("combo_to_points 需要至少一个靳三针组合实体。")
        combo_name = combos[0]
        cql = """
            MATCH (c:AcupointCombo {name: $combo_name})
            OPTIONAL MATCH (c)-[:HAS_POINT]->(stdPoint:Acupoint)
            OPTIONAL MATCH (c)-[:HAS_LOCAL_POINT]->(localPoint:ComboLocalPoint)
            OPTIONAL MATCH (c)-[:HAS_COMBO]->(subCombo:AcupointCombo)
            RETURN
                c.name AS combo_name,
                collect(DISTINCT stdPoint.name)   AS std_points,
                collect(DISTINCT localPoint.name) AS local_points,
                collect(DISTINCT subCombo.name)   AS sub_combos
            """
        params = {"combo_name": combo_name}
        return cql, run_cypher(cql, params)

    else:
        logger.warning(f"未知 query_type: {qt}")
        return "", []

# ==========================
# FastAPI 应用
# ==========================
app = FastAPI(
    title="Jin's Three-Needle KG+LLM QA System",
    description="基于靳三针知识图谱与 Qwen3 的智能问答后端。",
    version="1.0.0",
)


# 允许的前端来源
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]#可以修改

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # 只允许这两个源
    allow_credentials=True,
    allow_methods=["*"],          # 允许所有方法：GET, POST, ...
    allow_headers=["*"],          # 允许所有请求头
)

@app.post("/api/qa", response_model=QAResponse)
def qa_endpoint(req: QARequest):
    try:
        question = (req.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空。")

        # 1. 整理 history（只保留 user/assistant）
        history_for_llm: Optional[List[Dict[str, str]]] = None
        if req.history:
            history_for_llm = []
            for h in req.history:
                if h.role in ("user", "assistant"):
                    history_for_llm.append(
                        {"role": h.role, "content": h.content}
                    )

        # 2. LLM 解析 query_type + 实体
        parsed = call_llm_for_parse(question, history_for_llm)

        # 3. 实体对齐到图谱标准名称
        parsed_norm, entity_suggest = normalize_entities(parsed)
        query_type = parsed_norm["query_type"]
        entities = {
            "diseases": parsed_norm.get("diseases") or [],
            "combos": parsed_norm.get("combos") or [],
            "points": parsed_norm.get("points") or [],
        }
        use_kg = True if req.use_kg is None else bool(req.use_kg)

        # ==========================
        # 分支 1：不使用 KG（仅 LLM）
        # ==========================
        if not use_kg:
            ans = call_llm_for_answer_no_kg(
                question=question,
                query_type=query_type,
                entities=entities,
                history=history_for_llm,
            )
            return QAResponse(
                answer=ans,
                query_type=query_type,
                entities=entities,
                cypher=None,
                records=None,
                entity_suggest=entity_suggest or None,
            )
        # ==========================
        # 分支 2：使用 KG
        # ==========================
        # 4. unknown 直接回答
        if query_type == "unknown":
            ans = call_llm_for_answer(
                question=question,
                records=[],
                query_type=query_type,
                entities=entities,
                history=history_for_llm,
            )
            return QAResponse(
                answer=ans,
                query_type=query_type,
                entities=entities,
                cypher=None,
                records=[],
                entity_suggest=entity_suggest or None,
            )

        # 5. 构造 & 执行 Cypher（完全基于文中实体/关系）
        max_plans = req.options.max_plans if req.options else 10
        cypher, records = build_and_run_query(parsed_norm, max_plans=max_plans)

        # 6. 生成自然语言回答
        ans = call_llm_for_answer(
            question=question,
            records=records,
            query_type=query_type,
            entities=entities,
            history=history_for_llm,
        )

        return QAResponse(
            answer=ans,
            query_type=query_type,
            entities=entities,
            cypher=cypher or None,
            records=records,
            entity_suggest=entity_suggest or None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("QA 处理发生错误")
        raise HTTPException(status_code=500, detail=f"服务器内部错误：{e}")


@app.get("/health")
def health_check():
    return {"status": "ok"}