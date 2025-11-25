import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

from py2neo import Graph, Node, NodeMatcher, Relationship


class AcuKGBuilder:
    def __init__(self, uri: str, user: str, password: str):
        self.graph = Graph(uri, auth=(user, password))
        self.matcher = NodeMatcher(self.graph)

        # 缓存
        self.acupoint_by_name: Dict[str, Node] = {}
        self.acupoint_by_code: Dict[str, Node] = {}
        self.combo_by_name: Dict[str, Node] = {}
        self.disease_by_name: Dict[str, Node] = {}

        self._init_constraints()

    def _init_constraints(self):
        # 标准穴位
        self.graph.run("""
        CREATE CONSTRAINT acupoint_code_unique IF NOT EXISTS
        FOR (a:Acupoint) REQUIRE a.code IS UNIQUE
        """)
        self.graph.run("""
        CREATE CONSTRAINT acupoint_name_unique IF NOT EXISTS
        FOR (a:Acupoint) REQUIRE a.name IS UNIQUE
        """)
        # 组合
        self.graph.run("""
        CREATE CONSTRAINT combo_name_unique IF NOT EXISTS
        FOR (c:AcupointCombo) REQUIRE c.name IS UNIQUE
        """)
        # 疾病
        self.graph.run("""
        CREATE CONSTRAINT disease_name_unique IF NOT EXISTS
        FOR (d:Disease) REQUIRE d.name IS UNIQUE
        """)
        # 方案
        self.graph.run("""
        CREATE CONSTRAINT plan_id_unique IF NOT EXISTS
        FOR (p:TreatmentPlan) REQUIRE p.plan_id IS UNIQUE
        """)
        # 原始未标准化点名
        self.graph.run("""
        CREATE CONSTRAINT raw_point_name_unique IF NOT EXISTS
        FOR (r:RawPointName) REQUIRE r.name IS UNIQUE
        """)

    # =========================================================
    # 一、导入 GBT 标准穴位库
    # =========================================================

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
                    node = Node(
                        "Acupoint",
                        name=name,
                        code=code,
                        pinyin=pinyin,
                        meridian=meridian,
                        location=location
                    )
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

    # =========================================================
    # 二、导入靳三针组合库
    # =========================================================

    def import_jinsanzhen_combos_from_usage(self, usage_path: str):
        """

        预期每行 JSON 类似：
        {
          "point_group_name": "鼻三针",
          "indications": "用于各种急慢性鼻炎、鼻窦炎等……",
          "points": ["迎香", "上迎香", "印堂"],
          "acupuncture_method": {
              "迎香": "斜刺或平刺，0.3~0.5寸；向内上方斜刺0.5~1.2寸……",
              "上迎香": "向内上方斜刺0.3~0.5寸……",
              "印堂": "从上向下平刺0.3~0.5寸……"
          }
        }

        建模策略（**完全替代原组合 jsonl**）：
        - 为每个穴组建 AcupointCombo 节点：
            name = point_group_name
            indications = indications
            acupuncture_method_json = 整个 acupuncture_method 的 JSON 字符串
        - 对 points 中的每个名称 p：
            * 若是已存在的组合名 -> (combo)-[:HAS_COMBO {needle_method: ?}]->(subCombo)
            * 若是标准腧穴     -> (combo)-[:HAS_POINT {needle_method: "..."}]->(acupoint)
            * 否则视为局部点   -> (combo)-[:HAS_LOCAL_POINT {needle_method: "..."}]->(localPoint)
        - 针刺说明来自 acupuncture_method[p]，挂到边属性 needle_method 上。
        """
        path = Path(usage_path)
        assert path.exists(), f"Jin Sanzhen usage file not found: {path}"

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"无法解析靳三针使用 jsonl 中的一行：{line}\n错误：{e}"
                    )

                combo_name = (rec.get("point_group_name") or "").strip()
                if not combo_name:
                    continue

                indications = (rec.get("indications") or "").strip()
                points = rec.get("points") or []
                ac_m = rec.get("acupuncture_method") or {}

                # 1) 创建 / 获取组合节点
                combo_node = self._get_or_create_combo(combo_name)

                # 记录适应证 + 整体刺法 JSON
                updated = False
                if indications and combo_node.get("indications") != indications:
                    combo_node["indications"] = indications
                    updated = True
                if ac_m:
                    combo_node["acupuncture_method_json"] = json.dumps(
                        ac_m, ensure_ascii=False
                    )
                    updated = True
                if updated:
                    self.graph.push(combo_node)

                # 2) 逐个处理 points，并在边上附加针刺说明
                for raw_p in points:
                    p = (raw_p or "").strip()
                    if not p:
                        continue

                    # 尝试从 acupuncture_method 中取到对应刺法说明
                    method_text = ""
                    if isinstance(ac_m, dict):
                        method_text = (ac_m.get(p) or "").strip()

                    # 2.1 若 p 本身是另一个组合名（如 “晕痛针” 中包含 “四神针”）
                    sub_combo = self.matcher.match("AcupointCombo", name=p).first()
                    if sub_combo is not None:
                        rel = Relationship(combo_node, "HAS_COMBO", sub_combo)
                        if method_text:
                            rel["needle_method"] = method_text
                        self.graph.merge(rel)
                        continue

                    # 2.2 当作标准腧穴名
                    acup = self._find_acupoint_by_name(p)
                    if acup is not None:
                        rel = Relationship(combo_node, "HAS_POINT", acup)
                        if method_text:
                            rel["needle_method"] = method_text
                        self.graph.merge(rel)
                        continue

                    # 2.3 否则作为局部点（ComboLocalPoint）
                    local_node = self.matcher.match("ComboLocalPoint", name=p).first()
                    if local_node is None:
                        local_node = Node("ComboLocalPoint", name=p)
                        self.graph.merge(local_node, "ComboLocalPoint", "name")

                    rel = Relationship(combo_node, "HAS_LOCAL_POINT", local_node)
                    if method_text:
                        rel["needle_method"] = method_text
                    self.graph.merge(rel)

        print(f"[OK] Imported Jin Sanzhen combos from usage jsonl (with needle methods): {usage_path}")

    def _get_or_create_combo(self, name: str) -> Node:
        if name in self.combo_by_name:
            return self.combo_by_name[name]

        node = self.matcher.match("AcupointCombo", name=name).first()
        if node is None:
            node = Node("AcupointCombo", name=name, type="JinCombo")
            self.graph.merge(node, "AcupointCombo", "name")
        self.combo_by_name[name] = node
        return node

    def _link_combo_position(self, combo: Node, text: str):
        raw = (text or "").strip()
        if not raw:
            return

        # 1) 若本身就是另一个组合名（如 晕痛针 中包含 四神针）
        target_combo = self.matcher.match("AcupointCombo", name=raw).first()
        if target_combo is not None:
            rel = Relationship(combo, "HAS_COMBO", target_combo)
            self.graph.merge(rel)
            return

        # 2) 尝试括号里提取标准穴名
        bracket_name = self._extract_bracket_point_name(raw)
        if bracket_name:
            acup = self._find_acupoint_by_name(bracket_name)
            if acup is not None:
                rel = Relationship(combo, "HAS_POINT", acup)
                self.graph.merge(rel)
                return
            # 如果括号内不是标准穴，再尝试整个字符串

        # 3) 整段文本是否就是标准穴名
        acup = self._find_acupoint_by_name(raw)
        if acup is not None:
            rel = Relationship(combo, "HAS_POINT", acup)
            self.graph.merge(rel)
            return

        # 4) 否则作为局部点保存
        self._create_local_point(combo, raw)

    def _extract_bracket_point_name(self, s: str) -> Optional[str]:
        # 中文括号
        m = re.search(r"（([^（）]+)）", s)
        if m:
            inner = m.group(1).strip()
            # 若内部有多个，用顿号等分隔，取第一个
            for sep in ["，", "、", ",", ";"]:
                if sep in inner:
                    inner = inner.split(sep)[0].strip()
                    break
            return inner or None

        # 英文括号
        m = re.search(r"\(([^()]+)\)", s)
        if m:
            inner = m.group(1).strip()
            for sep in ["，", "、", ",", ";"]:
                if sep in inner:
                    inner = inner.split(sep)[0].strip()
                    break
            return inner or None

        return None

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

    def _create_local_point(self, combo: Node, raw_text: str):
        node = self.matcher.match("ComboLocalPoint", name=raw_text).first()
        if node is None:
            node = Node("ComboLocalPoint", name=raw_text)
            self.graph.merge(node, "ComboLocalPoint", "name")

        rel = Relationship(combo, "HAS_LOCAL_POINT", node)
        self.graph.merge(rel)

    # =========================================================
    # 三、导入 all_marked_merged.jsonl 方案
    # =========================================================

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

        # 导入结束后，生成疾病级汇总关系
        self._build_disease_summary_edges()
        print(f"[OK] Imported treatment plans from {plans_path}")

    def _get_or_create_disease(self, name: str) -> Node:
        if name in self.disease_by_name:
            return self.disease_by_name[name]

        node = self.matcher.match("Disease", name=name).first()
        if node is None:
            node = Node("Disease", name=name)
            self.graph.merge(node, "Disease", "name")
        self.disease_by_name[name] = node
        return node

    def _effect_level_from_text(self, effect: str) -> Optional[int]:
        """
        粗略标签：0 = 无效/差，1 = 有效，2 = 显著/痊愈
        可以后续再细化。
        """
        if not effect:
            return None
        text = effect

        # 明显好转、总有效率高等
        if any(k in text for k in ["痊愈", "显效", "显著", "疗效较好", "总有效率", "优于对照组"]):
            return 2
        if any(k in text for k in ["有效", "改善", "好转"]):
            return 1
        if any(k in text for k in ["无效", "效果不佳"]):
            return 0
        return None

    def _bool_from_method(self, text: str, keywords: List[str]) -> bool:
        if not text:
            return False
        return any(k in text for k in keywords)

    def _normalize_points_field(self, value: Any) -> List[str]:
        """
        main_points, auxiliary_points 现在已经是列表，多数情况直接返回。
        兼容如果后面你有其他数据源写成字符串。
        """
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

    def _import_single_plan(self, rec: Dict[str, Any], index: int):
        disease_name = (rec.get("disease") or "").strip()
        if not disease_name:
            return

        disease_node = self._get_or_create_disease(disease_name)

        # plan_id：简单用 disease + index
        plan_id = f"{disease_name}_{index:04d}"

        method = (rec.get("method") or "").strip()
        course = (rec.get("course") or "").strip()
        effect = (rec.get("effect") or "").strip()
        position = (rec.get("position") or "").strip()

        effect_level = self._effect_level_from_text(effect)
        has_ea = self._bool_from_method(method, ["电针", "脑电仿生电刺激"])
        has_mox = self._bool_from_method(method, ["艾灸", "温针灸", "艾条", "火针", "灸"])
        has_drug = self._bool_from_method(method + effect, ["汤", "片", "胶囊", "口服", "中药", "西药", "药物"])

        plan_props = {
            "disease": disease_name,
            "method_text": method,
            "course_text": course,
            "effect_text": effect,
            "position_text": position,
            "effect_level": effect_level,
            "has_electroacupuncture": has_ea,
            "has_moxibustion": has_mox,
            "has_drug": has_drug,
        }

        plan_node = self._get_or_create_plan(plan_id, plan_props)

        # Disease -> TreatmentPlan
        self.graph.merge(Relationship(disease_node, "HAS_PLAN", plan_node))

        # 主穴 / 配穴
        main_points = self._normalize_points_field(rec.get("main_points"))
        aux_points = self._normalize_points_field(rec.get("auxiliary_points"))

        self._link_points_for_plan(plan_node, main_points, is_main=True)
        self._link_points_for_plan(plan_node, aux_points, is_main=False)

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

    def _link_points_for_plan(self, plan_node: Node, points: List[str], is_main: bool):
        rel_type = "MAIN_POINT" if is_main else "AUX_POINT"

        for raw in points:
            name = raw.strip()
            if not name:
                continue

            # 1) 先看是不是组合（靳三针/其他组合）
            combo = self.combo_by_name.get(name)
            if combo is None:
                combo = self.matcher.match("AcupointCombo", name=name).first()
                if combo is not None:
                    self.combo_by_name[name] = combo
            if combo is not None:
                self.graph.merge(Relationship(plan_node, rel_type, combo))
                continue

            # 2) 再看是不是标准单穴
            acup = self._find_acupoint_by_name(name)
            if acup is not None:
                self.graph.merge(Relationship(plan_node, rel_type, acup))
                continue

            # 3) 诸如 “颞Ⅰ针”“言语一区”“焦氏头针一区”等特定分区：
            #    这些在 GBT 标准里没有，既不是组合，也不是标准穴，落在 RawPointName。
            raw_node = self.matcher.match("RawPointName", name=name).first()
            if raw_node is None:
                raw_node = Node("RawPointName", name=name)
                self.graph.merge(raw_node, "RawPointName", "name")
            self.graph.merge(Relationship(plan_node, rel_type, raw_node))

    # =========================================================
    # 四、疾病级别汇总 MAIN_POINT_SUMMARY / AUX_POINT_SUMMARY
    # =========================================================

    def _build_disease_summary_edges(self):
        cypher_main = """
        MATCH (d:Disease)-[:HAS_PLAN]->(p:TreatmentPlan)-[:MAIN_POINT]->(x)
        MERGE (d)-[:MAIN_POINT_SUMMARY]->(x)
        """
        cypher_aux = """
        MATCH (d:Disease)-[:HAS_PLAN]->(p:TreatmentPlan)-[:AUX_POINT]->(x)
        MERGE (d)-[:AUX_POINT_SUMMARY]->(x)
        """
        self.graph.run(cypher_main)
        self.graph.run(cypher_aux)
        print("[OK] Built disease-level summary edges")

    def clear_graph(self):
        """
        清空整个知识图谱数据库
        删除所有节点和关系
        """
        # 删除所有关系
        self.graph.run("MATCH ()-[r]->() DELETE r")

        # 删除所有节点及其属性
        self.graph.run("MATCH (n) DETACH DELETE n")

        # 清空缓存
        self.acupoint_by_name.clear()
        self.acupoint_by_code.clear()
        self.combo_by_name.clear()
        self.disease_by_name.clear()


        print("[OK] Graph cleared successfully")
if __name__ == "__main__":
    # 1. 修改为你的 Neo4j 配置
    URI = "bolt://localhost:7687"
    USER = "neo4j"
    PASSWORD = "Your PASSWORD"

    # 2. 修改为你三个文件的实际路径
    GBT_FILE = "GBT+12346-2021.jsonl"
    COMBO_FILE = "jinsanzhen_composition.jsonl"
    PLANS_FILE = "all_marked_merged.jsonl"


    builder = AcuKGBuilder(URI, USER, PASSWORD)
    builder.clear_graph()
    # 顺序：先标准穴位 → 再组合 → 再方案
    builder.import_gbt_points(GBT_FILE)
    builder.import_jinsanzhen_combos_from_usage(COMBO_FILE)

    builder.import_plans(PLANS_FILE)
