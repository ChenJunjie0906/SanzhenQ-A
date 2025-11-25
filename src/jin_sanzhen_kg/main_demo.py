import os
from .extract_text import PDFBatchExtractor
from .batch_auto_label import BatchAutoLabeler
from .build_graph import AcuKGBuilder
from .merge_dedup_labels import merge_jsonl  


def main():
    """
    靳三针知识图谱构建 demo 主程序
    在轻量数据集上串联：
    PDF 文本提取 → 自动结构化标注 → 标注结果合并过滤 → Neo4j 知识图谱构建
    """

    # ===============================
    # 0. 统一基于仓库根目录配置路径
    # ===============================
    # 当前文件: repo_root/src/jin_sanzhen_kg/main_demo.py
    # 仓库根目录 = 本文件向上三级
    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    # 数据目录（建议与你现在的 data 结构对应）
    PDF_FOLDER = os.path.join(BASE_DIR, "data", "raw", "pdf_demo")          # demo PDF
    TXT_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "interim", "extracted_texts_demo")     # OCR 结果
    JSONL_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "interim", "labeled_jsonl_demo")   # LLM 标注结果
    TEMP_PAGES_DIR = os.path.join(BASE_DIR, "data", "interim", "temp_pages")

    # Neo4j 连接从环境变量读取（不要硬编码密码）
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    if not NEO4J_PASSWORD:
        raise RuntimeError("请先设置环境变量 NEO4J_PASSWORD 再运行 main_demo.py")

    # 创建必要目录
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(JSONL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_PAGES_DIR, exist_ok=True)

    print("🚀 靳三针知识图谱构建 demo 流程启动")

    # ===============================
    # 1. PDF 文本提取阶段（demo）
    # ===============================
    print("\n📂 第一阶段：PDF 文本提取（demo）")

    extractor = PDFBatchExtractor(
        pdf_folder=PDF_FOLDER,
        txt_output_dir=TXT_OUTPUT_DIR,
        base_temp_dir=TEMP_PAGES_DIR,
        lang="ch",
        max_threads=1,   # demo 用单线程即可
        dpi=200,         # demo 可适当降低分辨率
    )

    # demo：约定只处理一个示例 PDF
    pdf_path = os.path.join(PDF_FOLDER, "example_jin_sanzhen.pdf")
    if os.path.exists(pdf_path):
        extractor.process_single_pdf(pdf_path)
    else:
        print(f"⚠️ 未找到 demo PDF 文件：{pdf_path}")
        print("   可以将任意 1 个示例 PDF 放在 data/raw/pdf_demo/ 下，并命名为 example_jin_sanzhen.pdf")
        print("   本次将跳过 OCR 阶段（如果你已经有手工准备的 txt 也没关系）。")

    # ===============================
    # 2. 自动标注阶段（demo）
    # ===============================
    print("\n🏷️ 第二阶段：自动结构化标注（demo）")

    labeler = BatchAutoLabeler(
        txt_dir=TXT_OUTPUT_DIR,
        out_dir=JSONL_OUTPUT_DIR,
        max_workers=1,  # demo 不需要开很多线程
    )

    # demo：统一按“疾病治疗信息”提示词来抽取
    labeler.batch_process(prompt_type="disease")

    # ===============================
    # 3. 合并标注结果阶段（demo）
    # ===============================
    print("\n🔄 第三阶段：合并 & 过滤标注结果（demo）")

    # 这里直接调用 merge_jsonl()，其内部 IN_DIR/OUT_FILE 可以暂时沿用你原来的设置
    merge_jsonl()

    merged_plans_file = os.path.join(BASE_DIR, "all_marked_merged.jsonl")
    if not os.path.exists(merged_plans_file):
        print(f"⚠️ 未在 {BASE_DIR} 找到合并后的 all_marked_merged.jsonl，"
              "请检查 merge_dedup_labels.py 中 IN_DIR/OUT_FILE 设置。")
        return

    # ===============================
    # 4. 知识图谱构建阶段（demo）
    # ===============================
    print("\n🧠 第四阶段：知识图谱构建（demo）")

    builder = AcuKGBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # demo 建议每次清空图谱，保持可重复构建
    builder.clear_graph()

    # 标准穴位 & 组合 & 方案，使用 demo 版 jsonl
    processed_dir = os.path.join(BASE_DIR, "data", "raw", "processed")
    gbt_file = os.path.join(processed_dir, "GBT+12346-2021_demo.jsonl")
    combo_file = os.path.join(processed_dir, "jinsanzhen_usage_demo.jsonl")
    plans_file = merged_plans_file  # 上一步 merge 的输出

    # 导入标准穴位库（demo 小样本）
    if os.path.exists(gbt_file):
        builder.import_gbt_points(gbt_file)
    else:
        print(f"⚠️ 警告：未找到标准穴位 demo 文件 {gbt_file}")

    # 导入靳三针组合库（demo 小样本）
    if os.path.exists(combo_file):
        builder.import_jinsanzhen_combos_from_usage(combo_file)
    else:
        print(f"⚠️ 警告：未找到靳三针组合 demo 文件 {combo_file}")

    # 导入标注后的治疗方案（由 demo OCR+标注+合并得到）
    if os.path.exists(plans_file):
        builder.import_plans(plans_file)
    else:
        print(f"⚠️ 警告：未找到治疗方案文件 {plans_file}")

    print("\n🎉 demo 流程已完成")
    print("   - OCR 文本目录:", TXT_OUTPUT_DIR)
    print("   - LLM 标注结果目录:", JSONL_OUTPUT_DIR)
    print("   - 合并后方案文件:", merged_plans_file)
    print("   - Neo4j 中已构建一个小规模示例知识图谱（可在浏览器中连接查看）")


if __name__ == "__main__":
    main()

