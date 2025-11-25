import os
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import glob


class PDFBatchExtractor:
    def __init__(self, pdf_folder="pdf_files", txt_output_dir="extracted_texts",
                 base_temp_dir="temp_pages", lang="ch", max_threads=2, dpi=300):
        """
        初始化批量PDF文本提取器

        Args:
            pdf_folder: PDF文件所在文件夹
            txt_output_dir: 文本输出文件夹
            base_temp_dir: 基础临时文件夹
            lang: OCR语言设置
            max_threads: 并行处理线程数
            dpi: 图像分辨率
        """
        self.pdf_folder = pdf_folder
        self.txt_output_dir = txt_output_dir
        self.base_temp_dir = base_temp_dir
        self.lang = lang
        self.max_threads = max_threads
        self.dpi = dpi

        # 创建必要的目录
        if not os.path.exists(self.txt_output_dir):
            os.makedirs(self.txt_output_dir)
        if not os.path.exists(self.base_temp_dir):
            os.makedirs(self.base_temp_dir)
        if not os.path.exists(self.pdf_folder):
            os.makedirs(self.pdf_folder)

        # 初始化OCR引擎
        self.ocr = PaddleOCR(use_textline_orientation=True, lang=self.lang)

    def process_page(self, page_num, image_file):
        """
        处理单页OCR识别

        Args:
            page_num: 页码
            image_file: 图片文件路径

        Returns:
            str: 识别出的文本内容
        """
        try:
            result = self.ocr.ocr(image_file)
            text_lines = []
            for line in result[0]:
                txt = line[1][0].strip()
                if txt:
                    text_lines.append(txt)
            page_text = "\n".join(text_lines)
            return f"\n\n📘【第 {page_num} 页】\n{page_text}\n"
        except Exception as e:
            return f"\n\n📘【第 {page_num} 页】\n[识别出错]: {e}\n"

    def process_single_pdf(self, pdf_path):
        """
        处理单个PDF文件

        Args:
            pdf_path: PDF文件路径
        """
        # 获取PDF文件名（不含扩展名）
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        # 创建以PDF命名的临时子文件夹
        temp_dir = os.path.join(self.base_temp_dir, pdf_name)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # PDF转图片
        print(f"📄 正在处理: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=self.dpi, fmt="png", output_folder=temp_dir)
        print(f"✅ 已生成 {len(images)} 页图片到 {temp_dir}")

        # 获取图片文件列表
        image_files = sorted([
            os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
            if f.endswith(".png")
        ])

        # OCR识别
        print("🤖 开始OCR识别...")
        results = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            tasks = {}
            for idx, f in enumerate(image_files):
                page_num = idx + 1
                tasks[executor.submit(self.process_page, page_num, f)] = page_num

            for future in tqdm(as_completed(tasks), total=len(tasks)):
                results.append(future.result())

        # 保存文本结果
        output_txt = os.path.join(self.txt_output_dir, f"{pdf_name}.txt")
        print("💾 正在合并文字输出...")
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(results))

        print(f"✅ OCR完成，结果已保存到: {output_txt}")

    def process_all_pdfs(self):
        """
        批量处理所有PDF文件
        """
        # 查找所有PDF文件
        pdf_files = glob.glob(os.path.join(self.pdf_folder, "*.pdf"))

        if not pdf_files:
            print("❌ 未找到PDF文件")
            return

        # 处理每个PDF文件
        for pdf_file in pdf_files:
            try:
                self.process_single_pdf(pdf_file)
            except Exception as e:
                print(f"❌ 处理 {pdf_file} 时出错: {e}")


# 使用示例
if __name__ == "__main__":
    extractor = PDFBatchExtractor()
    #extractor.process_all_pdfs()
    #若要处理单个PDF文件，打开下面的代码并修改文件路径
    extractor.process_single_pdf("GBT+12346-2021.pdf")