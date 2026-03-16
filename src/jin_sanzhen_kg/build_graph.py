# build_graph.py  ── 含穴位 & 疾病归一化

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

from py2neo import Graph, Node, NodeMatcher, Relationship

# ================================================================
#  归 一 化 配 置（按需增删）
# ================================================================

# -------- 单穴别名 → GBT 标准 name --------
POINT_ALIAS: Dict[str, str] = {
    "人中": "水沟",
    # 如 GBT 库中已有 "太阳"（EX-HN5）则不需要此行
    # "太阳": "太阳",
}

# -------- 穴组别名 → 已有穴组 name --------
COMBO_ALIAS: Dict[str, str] = {
    # 确认「挛三针」等同「足挛三针」后取消注释：
    # "挛三针": "足挛三针",
}

# -------- 需自动建为 AcupointCombo(type=ExtendedCombo) 的名称 --------
NEW_COMBOS = {
    "下肢挛三针", "上肢挛三针", "挛三针",
    "膝三针", "醒神针", "眠三针",
    "颞上三针", "左颞上三针",
    "四关穴", "四关",
}

# -------- 疾病同义词 → 规范名 --------
DISEASE_ALIAS: Dict[str, str] = {
    # 中风
    "脑卒中": "中风", "中风（脑卒中）": "中风", "中风后遗症": "中风",
    # 中风后偏瘫
    "脑卒中偏瘫": "中风后偏瘫",
    "中风后偏瘫（迟缓性/痉挛性）": "中风后偏瘫",
    "缺血性中风后痉挛性偏瘫": "中风后偏瘫",
    "卒中后运动障碍": "中风后偏瘫",
    "脑梗死下肢功能障碍": "中风后偏瘫",
    # 中风后假性球麻痹
    "脑卒中假性球麻痹": "中风后假性球麻痹",
    "中风性假性球麻痹": "中风后假性球麻痹",
    # 吞咽障碍
    "吞咽功能障碍": "吞咽障碍",
    # 中风后肩手综合征
    "脑卒中后肩手综合征": "中风后肩手综合征",
    # 卒中后认知障碍
    "中风后认知功能障碍": "卒中后认知障碍",
    # 中风后抑郁
    "中风后抑郁症": "中风后抑郁",
    # 孤独症谱系障碍
    "儿童孤独症": "孤独症谱系障碍", "自闭症": "孤独症谱系障碍",
    "儿童自闭症": "孤独症谱系障碍", "小儿自闭症": "孤独症谱系障碍",
    "孤独症": "孤独症谱系障碍",
    # 自闭症语言障碍
    "自闭症患儿语言发育障碍": "自闭症语言障碍",
    # 精神发育迟缓
    "小儿精神发育迟缓": "精神发育迟缓", "智力低下": "精神发育迟缓",
    "精神发育迟滞": "精神发育迟缓", "儿童精神发育迟滞": "精神发育迟缓",
    "智力障碍": "精神发育迟缓", "儿童智力障碍": "精神发育迟缓",
    "智力发育障碍儿童": "精神发育迟缓",
    "儿童精神发育迟滞（弱智）": "精神发育迟缓",
    # 脑瘫
    "脑性瘫痪": "脑瘫", "小儿脑瘫": "脑瘫",
    "脑性瘫痪高危儿": "脑瘫高危儿",
    # 痉挛型脑瘫
    "痉挛型脑性瘫痪": "痉挛型脑瘫", "小儿痉挛型脑瘫": "痉挛型脑瘫",
    # 失眠
    "不寐": "失眠", "失眠症": "失眠", "失眠症（另一方案）": "失眠",
    # 面瘫
    "周围性面瘫": "面瘫", "顽固性面瘫": "面瘫", "急性期周围性面瘫": "面瘫",
    # 过敏性鼻炎
    "变应性鼻炎": "过敏性鼻炎", "变应性／过敏性鼻炎": "过敏性鼻炎",
    # 偏头痛
    "无先兆型偏头痛": "偏头痛",
    # 紧张型头痛
    "紧张性头痛": "紧张型头痛",
    # 耳鸣耳聋
    "耳鸣、耳聋": "耳鸣耳聋",
    # 肥胖
    "单纯性肥胖": "单纯性肥胖症", "肥胖症": "单纯性肥胖症",
    # 肩周炎
    "肩关节周围炎": "肩周炎",
    # 抽动秽语综合征
    "小儿多发性抽动症": "抽动秽语综合征", "小儿抽动症": "抽动秽语综合征",
    "抽动-秽语综合征（多发性抽动症）": "抽动秽语综合征",
    # 注意缺陷多动障碍
    "儿童多动症": "注意缺陷多动障碍",
    "注意缺陷多动障碍（ADHD）": "注意缺陷多动障碍",
    # 全面发育迟缓
    "全面性发育迟缓": "全面发育迟缓",
    # 痴呆
    "老年痴呆症": "痴呆",
    # 癫痫
    "小儿癫痫": "癫痫", "儿童痫证（癫痫）": "癫痫",
    # 膝骨关节炎
    "膝关节骨性关节炎": "膝骨关节炎", "膝骨性关节炎": "膝骨关节炎",
    "老年膝骨关节炎": "膝骨关节炎",
    # 高脂血症
    "高脂血症（痰浊阻遏型）": "高脂血症",
    # 帕金森抑郁
    "帕金森轻中度抑郁": "帕金森合并抑郁",
}

# -------- 疾病分类（规范名 → 大类） --------
DISEASE_CATEGORY: Dict[str, str] = {
    # 脑血管病
    "中风": "脑血管病", "中风后偏瘫": "脑血管病", "中风后抑郁": "脑血管病",
    "中风后假性球麻痹": "脑血管病", "中风后肩手综合征": "脑血管病",
    "中风后下肢痉挛": "脑血管病", "中风失语症": "脑血管病",
    "中风先兆": "脑血管病", "卒中后认知障碍": "脑血管病",
    "吞咽障碍": "脑血管病", "血管性痴呆": "脑血管病",
    "颅脑损伤后非流畅性失语": "脑血管病",
    "脑外伤术后认知功能障碍": "脑血管病",
    # 儿童发育障碍
    "孤独症谱系障碍": "儿童发育障碍", "自闭症语言障碍": "儿童发育障碍",
    "精神发育迟缓": "儿童发育障碍",
    "脑瘫": "儿童发育障碍", "脑瘫高危儿": "儿童发育障碍",
    "痉挛型脑瘫": "儿童发育障碍", "不随意运动型脑瘫": "儿童发育障碍",
    "痉挛型偏瘫": "儿童发育障碍",
    "全面发育迟缓": "儿童发育障碍",
    "全面发育迟缓伴孤独症": "儿童发育障碍",
    "运动发育指标延迟": "儿童发育障碍", "儿童语言障碍": "儿童发育障碍",
    # 儿童行为障碍
    "注意缺陷多动障碍": "儿童行为障碍", "抽动秽语综合征": "儿童行为障碍",
    # 神经精神疾病
    "失眠": "神经精神疾病", "围绝经期失眠": "神经精神疾病",
    "心脾两虚型失眠": "神经精神疾病",
    "痴呆": "神经精神疾病", "阿尔茨海默病": "神经精神疾病",
    "轻度认知功能障碍": "神经精神疾病",
    "癫痫": "神经精神疾病", "精神分裂症": "神经精神疾病",
    "郁证": "神经精神疾病", "广泛性焦虑症": "神经精神疾病",
    "帕金森合并抑郁": "神经精神疾病",
    "眩晕": "神经精神疾病", "颤证": "神经精神疾病",
    # 疼痛
    "偏头痛": "疼痛", "紧张型头痛": "疼痛", "颈源性头痛": "疼痛",
    "三叉神经痛": "疼痛", "原发性痛经": "疼痛", "胁痛": "疼痛",
    # 面神经疾病
    "面瘫": "面神经疾病",
    # 骨伤科
    "颈椎病": "骨伤科", "颈型颈椎病": "骨伤科",
    "椎动脉型颈椎病": "骨伤科", "气滞血瘀型颈型颈椎病": "骨伤科",
    "肩周炎": "骨伤科", "肩峰下撞击综合征": "骨伤科",
    "腰椎间盘突出症": "骨伤科", "腰痛": "骨伤科",
    "腰椎间盘突出症所致坐骨神经痛": "骨伤科",
    "膝骨关节炎": "骨伤科", "膝踝骨关节炎": "骨伤科",
    "踝关节骨性关节炎": "骨伤科",
    "类风湿性关节炎": "骨伤科", "痛风性关节炎慢性期": "骨伤科",
    "落枕": "骨伤科", "项背肌筋膜炎": "骨伤科",
    "放射性颞下颌关节强直": "骨伤科",
    # 五官科
    "过敏性鼻炎": "五官科", "耳鸣耳聋": "五官科",
    "神经性耳鸣": "五官科", "神经性耳聋": "五官科",
    "全聋型突发性感音神经性聋": "五官科",
    "眼科疾病": "五官科", "视神经萎缩": "五官科",
    "外展神经麻痹": "五官科",
    # 代谢内分泌
    "单纯性肥胖症": "代谢内分泌", "高脂血症": "代谢内分泌",
    "糖尿病周围神经病变": "代谢内分泌", "月经不调": "代谢内分泌",
    # 其他
    "哮喘": "其他", "急性荨麻疹": "其他",
    "脾虚湿蕴型亚急性湿疹": "其他", "胆汁反流性胃炎": "其他",
    "慢性疲劳综合征": "其他", "运动性疲劳": "其他", "顽固性呃逆": "其他",
}


# ================================================================
#  Builder
# ================================================================

class AcuKGBuilder:
    def __init__(self, uri: str, user: str, password: str):
        self.graph = Graph(uri, auth=(user, password))
        self.matcher = NodeMatcher(self.graph)

        self.acupoint_by_name: Dict[str, Node] = {}
        self.acupoint_by_code: Dict[str, Node] = {}
        self.combo_by_name: Dict[str, Node] = {}
        self.disease_by_name: Dict[str, Node] = {}

        self._init_constraints()

    # -------------------- 约束 --------------------
    def _init_constraints(self):
        for cypher in [
            "CREATE CONSTRAINT acupoint_code_unique IF NOT EXISTS FOR (a:Acupoint) REQUIRE a.code IS UNIQUE",
            "CREATE CONSTRAINT acupoint_name_unique IF NOT EXISTS FOR (a:Acupoint) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT combo_name_unique IF NOT EXISTS FOR (c:AcupointCombo) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT disease_name_unique IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT plan_id_unique IF NOT EXISTS FOR (p:TreatmentPlan) REQUIRE p.plan_id IS UNIQUE",
            "CREATE CONSTRAINT raw_point_name_unique IF NOT EXISTS FOR (r:RawPointName) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT disease_cat_unique IF NOT EXISTS FOR (dc:DiseaseCategory) REQUIRE dc.name IS UNIQUE",
        ]:
            self.graph.run(cypher)

    # =============================================================
    #  一、GBT 标准穴位库
    # =============================================================
    def import_gbt_points(self, gbt_path: str):
        path = Path(gbt_path)
        assert path.exists(), f"GBT file not found: {path}"
        tx = self.graph.begin()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                name = rec["point_name"].strip()
                code = rec["standard_code"].strip()
                pinyin = (rec.get("pinyin") or "").strip()
                meridian = (rec.get("meridian") or "").strip()
                location = (rec.get("location") or "").strip()

                node = self.matcher.match("Acupoint", code=code).first()
                if node is None:
                    node = Node("Acupoint", name=name, code=code,
                                pinyin=pinyin, meridian=meridian, location=location)
                    tx.merge(node, "Acupoint", "code")
                else:
                    node["name"] = name
                    node["pinyin"] = pinyin
                    node["meridian"] = meridian
                    node["location"] = location
                    tx.push(node)

                self.acupoint_by_code[code] = node
                self.acupoint_by_name[name] = node
        tx.commit()
        print(f"[OK] Imported GBT points from {gbt_path}")

    # =============================================================
    #  二、靳三针穴组
    # =============================================================
    def import_jinsanzhen_combos_from_usage(self, usage_path: str):
        path = Path(usage_path)
        assert path.exists(), f"File not found: {path}"
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON parse error: {line}\n{e}")

                combo_name = (rec.get("point_group_name") or "").strip()
                if not combo_name:
                    continue

                indications = (rec.get("indications") or "").strip()
                points = rec.get("points") or []
                ac_m = rec.get("acupuncture_method") or {}

                combo_node = self._get_or_create_combo(combo_name)
                updated = False
                if indications and combo_node.get("indications") != indications:
                    combo_node["indications"] = indications
                    updated = True
                if ac_m:
                    combo_node["acupuncture_method_json"] = json.dumps(ac_m, ensure_ascii=False)
                    updated = True
                if updated:
                    self.graph.push(combo_node)

                for raw_p in points:
                    p = (raw_p or "").strip()
                    if not p:
                        continue
                    method_text = ""
                    if isinstance(ac_m, dict):
                        method_text = (ac_m.get(p) or "").strip()

                    sub_combo = self.matcher.match("AcupointCombo", name=p).first()
                    if sub_combo is not None:
                        rel = Relationship(combo_node, "HAS_COMBO", sub_combo)
                        if method_text:
                            rel["needle_method"] = method_text
                        self.graph.merge(rel)
                        continue

                    acup = self._find_acupoint_by_name(p)
                    if acup is not None:
                        rel = Relationship(combo_node, "HAS_POINT", acup)
                        if method_text:
                            rel["needle_method"] = method_text
                        self.graph.merge(rel)
                        continue

                    local_node = self.matcher.match("ComboLocalPoint", name=p).first()
                    if local_node is None:
                        local_node = Node("ComboLocalPoint", name=p)
                        self.graph.merge(local_node, "ComboLocalPoint", "name")
                    rel = Relationship(combo_node, "HAS_LOCAL_POINT", local_node)
                    if method_text:
                        rel["needle_method"] = method_text
                    self.graph.merge(rel)

        print(f"[OK] Imported Jin-Sanzhen combos: {usage_path}")

    def _get_or_create_combo(self, name: str,
                              combo_type: str = "JinCombo") -> Node:
        if name in self.combo_by_name:
            return self.combo_by_name[name]
        node = self.matcher.match("AcupointCombo", name=name).first()
        if node is None:
            node = Node("AcupointCombo", name=name, type=combo_type)
            self.graph.merge(node, "AcupointCombo", "name")
        self.combo_by_name[name] = node
        return node

    def _find_acupoint_by_name(self, name: str) -> Optional[Node]:
        name = name.strip()
        if not name:
            return None
        if name in self.acupoint_by_name:
            return self.acupoint_by_name[name]
        node = self.matcher.match("Acupoint", name=name).first()
        if node is not None:
            self.acupoint_by_name[name] = node
        return node

    # =============================================================
    #  三、方案导入（含穴位 & 疾病归一化）★ 核心修改 ★
    # =============================================================
    def import_plans(self, plans_path: str):
        path = Path(plans_path)
        assert path.exists(), f"Plans file not found: {path}"
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._import_single_plan(rec, idx)

        self._build_disease_summary_edges()
        self._build_disease_categories()
        print(f"[OK] Imported treatment plans from {plans_path}")

    # ---------- 穴位名归一化 ----------
    @staticmethod
    def _normalize_point_name(raw: str):
        """
        '足三里（气虚血瘀）' → ('足三里', '气虚血瘀')
        '手三针（伴上肢瘫）' → ('手三针', '伴上肢瘫')
        '足三里'             → ('足三里', '')
        """
        raw = raw.strip()
        m = re.match(r'^(.+?)（(.+?)）$', raw)
        if not m:
            m = re.match(r'^(.+?)\((.+?)\)$', raw)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return raw, ""

    # ---------- 疾病名归一化 ----------
    def _get_or_create_disease(self, name: str) -> Node:
        canonical = DISEASE_ALIAS.get(name, name)
        if canonical in self.disease_by_name:
            return self.disease_by_name[canonical]
        node = self.matcher.match("Disease", name=canonical).first()
        if node is None:
            node = Node("Disease", name=canonical)
            cat = DISEASE_CATEGORY.get(canonical)
            if cat:
                node["category"] = cat
            self.graph.merge(node, "Disease", "name")
        self.disease_by_name[canonical] = node
        return node

    # ---------- 单条方案 ----------
    def _import_single_plan(self, rec: Dict[str, Any], index: int):
        raw_disease = (rec.get("disease") or "").strip()
        if not raw_disease:
            return

        disease_name = DISEASE_ALIAS.get(raw_disease, raw_disease)
        disease_node = self._get_or_create_disease(disease_name)

        plan_id = f"{disease_name}_{index:04d}"

        main_position_method = (rec.get("main_position_method") or "").strip()
        aux_position_method = (rec.get("aux_position_method") or "").strip()
        course = (rec.get("course") or "").strip()
        effect = (rec.get("effect") or "").strip()
        literature_source = (rec.get("literature_source") or "").strip()
        effect_level = self._effect_level_from_text(effect)

        plan_props = {
            "disease": disease_name,
            "disease_original": raw_disease,
            "main_position_method": main_position_method,
            "aux_position_method": aux_position_method,
            "course_text": course,
            "effect_text": effect,
            "effect_level": effect_level,
            "literature_source": literature_source,
        }
        plan_node = self._get_or_create_plan(plan_id, plan_props)
        self.graph.merge(Relationship(disease_node, "HAS_PLAN", plan_node))

        main_points = self._normalize_points_field(rec.get("main_points"))
        aux_points = self._normalize_points_field(rec.get("auxiliary_points"))
        self._link_points_for_plan(plan_node, main_points, is_main=True)
        self._link_points_for_plan(plan_node, aux_points, is_main=False)

    # ---------- ★ 关键：带归一化的穴位关联 ★ ----------
    def _link_points_for_plan(self, plan_node: Node,
                               points: List[str], is_main: bool):
        rel_type = "MAIN_POINT" if is_main else "AUX_POINT"

        for raw in points:
            raw = raw.strip()
            if not raw:
                continue

            # 1) 拆括号 → base + syndrome
            base, syndrome = self._normalize_point_name(raw)

            # 2) 单穴别名
            base = POINT_ALIAS.get(base, base)

            # 3) 穴组匹配（含 COMBO_ALIAS）
            combo_name = COMBO_ALIAS.get(base, base)
            combo = self.combo_by_name.get(combo_name)
            if combo is None:
                combo = self.matcher.match("AcupointCombo", name=combo_name).first()
                if combo is not None:
                    self.combo_by_name[combo_name] = combo
            if combo is not None:
                rel = Relationship(plan_node, rel_type, combo)
                if syndrome:
                    rel["syndrome"] = syndrome
                if combo_name != base:
                    rel["original_name"] = base
                self.graph.merge(rel)
                continue

            # 4) 高频非标穴组 → 自动建 ExtendedCombo
            if base in NEW_COMBOS:
                ext = self._get_or_create_combo(base, combo_type="ExtendedCombo")
                rel = Relationship(plan_node, rel_type, ext)
                if syndrome:
                    rel["syndrome"] = syndrome
                self.graph.merge(rel)
                continue

            # 5) GBT 标准穴位
            acup = self._find_acupoint_by_name(base)
            if acup is not None:
                rel = Relationship(plan_node, rel_type, acup)
                if syndrome:
                    rel["syndrome"] = syndrome
                self.graph.merge(rel)
                continue

            # 6) 仍无法识别 → RawPointName（用 base 作 name）
            raw_node = self.matcher.match("RawPointName", name=base).first()
            if raw_node is None:
                raw_node = Node("RawPointName", name=base)
                self.graph.merge(raw_node, "RawPointName", "name")
            rel = Relationship(plan_node, rel_type, raw_node)
            if syndrome:
                rel["syndrome"] = syndrome
            self.graph.merge(rel)

    # ---------- 辅助 ----------
    def _effect_level_from_text(self, effect: str) -> Optional[int]:
        if not effect:
            return None
        if any(k in effect for k in ["痊愈", "显效", "显著", "疗效较好", "总有效率", "优于对照组"]):
            return 2
        if any(k in effect for k in ["有效", "改善", "好转"]):
            return 1
        if any(k in effect for k in ["无效", "效果不佳"]):
            return 0
        return None

    def _normalize_points_field(self, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            s = value
            for sep in ["，", ",", "、", ";", " "]:
                s = s.replace(sep, " ")
            return [item.strip() for item in s.split() if item.strip()]
        return []

    def _get_or_create_plan(self, plan_id: str, props: Dict[str, Any]) -> Node:
        node = self.matcher.match("TreatmentPlan", plan_id=plan_id).first()
        if node is None:
            node = Node("TreatmentPlan", plan_id=plan_id, **props)
            self.graph.merge(node, "TreatmentPlan", "plan_id")
        else:
            for k, v in props.items():
                if v not in (None, "", []):
                    node[k] = v
            self.graph.push(node)
        return node

    # =============================================================
    #  四、汇总边 & 疾病分类
    # =============================================================
    def _build_disease_summary_edges(self):
        self.graph.run("""
            MATCH (d:Disease)-[:HAS_PLAN]->(p:TreatmentPlan)-[:MAIN_POINT]->(x)
            MERGE (d)-[:MAIN_POINT_SUMMARY]->(x)
        """)
        self.graph.run("""
            MATCH (d:Disease)-[:HAS_PLAN]->(p:TreatmentPlan)-[:AUX_POINT]->(x)
            MERGE (d)-[:AUX_POINT_SUMMARY]->(x)
        """)
        print("[OK] Built disease-level summary edges")

    def _build_disease_categories(self):
        """为每个 Disease 建立 DiseaseCategory 层"""
        for disease_name, category in DISEASE_CATEGORY.items():
            d_node = self.matcher.match("Disease", name=disease_name).first()
            if d_node is None:
                continue
            cat_node = self.matcher.match("DiseaseCategory", name=category).first()
            if cat_node is None:
                cat_node = Node("DiseaseCategory", name=category)
                self.graph.merge(cat_node, "DiseaseCategory", "name")
            self.graph.merge(Relationship(d_node, "BELONGS_TO", cat_node))
        print("[OK] Built disease category layer")

    # =============================================================
    #  清空
    # =============================================================
    def clear_graph(self):
        self.graph.run("MATCH ()-[r]->() DELETE r")
        self.graph.run("MATCH (n) DETACH DELETE n")
        self.acupoint_by_name.clear()
        self.acupoint_by_code.clear()
        self.combo_by_name.clear()
        self.disease_by_name.clear()
        print("[OK] Graph cleared")


# ================================================================
if __name__ == "__main__":
    URI = "bolt://localhost:7687"
    USER = "neo4j"
    PASSWORD = "Jacky@0906"

    GBT_FILE = "GBT+12346-2021.jsonl"
    COMBO_FILE = "靳三针穴组使用.jsonl"
    PLANS_FILE = "all_marked_merged_new（附文献）_corrected_v2.jsonl"

    builder = AcuKGBuilder(URI, USER, PASSWORD)
    builder.clear_graph()
    builder.import_gbt_points(GBT_FILE)
    builder.import_jinsanzhen_combos_from_usage(COMBO_FILE)
    builder.import_plans(PLANS_FILE)
