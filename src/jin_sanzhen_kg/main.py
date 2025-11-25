import os
from extract_text import PDFBatchExtractor
from batch_auto_label import BatchAutoLabeler
from build_graph import AcuKGBuilder
from merge_dedup_labels import LabelMerger


def main():
    """
    靳三针知识图谱构建主程序
    整合PDF文本提取、自动标注和知识图谱构建全流程
    """

    # ===============================
    # 配置参数
    # ===============================
    PDF_FOLDER = "pdf_files"  # PDF源文件目录
    TXT_OUTPUT_DIR = "extracted_texts"  # PDF提取文本输出目录
    JSONL_OUTPUT_DIR = "labeled_jsonl"  # 结构化标注结果输出目录
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "Jacky@0906"

    # 创建必要目录
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(JSONL_OUTPUT_DIR, exist_ok=True)

    print("🚀 靳三针知识图谱构建系统启动")

    # ===============================
    # 1. PDF文本提取阶段
    # ===============================
    print("\n📂 第一阶段：PDF文本提取")
    extractor = PDFBatchExtractor(
        pdf_folder=PDF_FOLDER,
        txt_output_dir=TXT_OUTPUT_DIR,
        base_temp_dir="temp_pages",
        lang="ch",
        max_threads=3,
        dpi=300
    )

    # 处理单个PDF文件
    pdf_path = os.path.join("pdf_files", "靳三针疗法流派临床经验全图解.pdf")
    #extractor.process_single_pdf(pdf_path)

    # ===============================
    # 2. 自动标注阶段
    # ===============================
    print("\n🏷️ 第二阶段：自动结构化标注")
    labeler = BatchAutoLabeler(
        txt_dir=TXT_OUTPUT_DIR,
        out_dir=JSONL_OUTPUT_DIR,
        max_workers=3
    )

    # 对单个文件进行标注（疾病治疗信息），
    #labeler.label_single_txt("extracted_texts/靳三针疗法流派临床经验全图解.txt", prompt_type="disease")

    # ===============================
    # 3. 合并标注结果阶段
    # ===============================
    print("\n🔄 第三阶段：合并标注结果")
    merger = LabelMerger(input_dir=JSONL_OUTPUT_DIR, output_file="all_marked_merged.jsonl")
    if not merger.merge():
        print("⚠️ 合并过程出现错误，请检查标注文件")
        return

    # ===============================
    # 4. 知识图谱构建阶段
    # ===============================
    print("\n🧠 第四阶段：知识图谱构建")

    # 初始化知识图谱构建器
    builder = AcuKGBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # 清空现有图谱（可选，用于重新构建）
    builder.clear_graph()

    # 导入标准穴位库
    gbt_file = "GBT+12346-2021.jsonl"
    if os.path.exists(gbt_file):
        builder.import_gbt_points(gbt_file)
    else:
        print(f"⚠️ 警告：未找到标准穴位文件 {gbt_file}")

    # 导入靳三针组合库
    combo_file = "靳三针穴组使用.jsonl"
    if os.path.exists(combo_file):
        builder.import_jinsanzhen_combos_from_usage(combo_file)
    else:
        print(f"⚠️ 警告：未找到靳三针组合文件 {combo_file}")

    # 导入标注后的治疗方案
    plans_file = "all_marked_merged.jsonl"
    if os.path.exists(plans_file):
        builder.import_plans(plans_file)
    else:
        print(f"⚠️ 警告：未找到治疗方案文件 {plans_file}")

    print("\n🎉 所有处理完成！")
    #print(f"📝 提取文本已保存至: {TXT_OUTPUT_DIR}")
    #print(f"📊 结构化数据已保存至: {JSONL_OUTPUT_DIR}")
    print(f"🌐 知识图谱已构建完成")


if __name__ == "__main__":
    main()
