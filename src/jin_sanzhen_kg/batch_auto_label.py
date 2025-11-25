import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class BatchAutoLabeler:
    """
    批量自动标注工具类：读取 extracted_texts 中的 txt，
    调用通义 Qwen 模型输出结构化 JSON，并写入 labeled_jsonl 目录。
    """

    def __init__(self, txt_dir="extracted_texts", out_dir="labeled_jsonl", max_workers=3):
        """
        初始化参数

        Args:
            txt_dir (str): 输入文本目录路径
            out_dir (str): 输出JSONL文件目录路径
            max_workers (int): 最大线程数
        """
        # 环境初始化
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 参数设置
        self.TXT_DIR = txt_dir
        self.OUT_DIR = out_dir
        self.MAX_WORKERS = max_workers

        # 创建输出目录
        os.makedirs(self.OUT_DIR, exist_ok=True)

    def build_prompt_for_disease_treatment(self, content: str) -> str:
        """
        构造用于疾病治疗信息提取的Prompt
        """
        return f"""
你是一位资深中医研究员，请对以下文献内容
进行结构化信息标注，并输出为标准 JSON。

需要提取字段：
- 疾病 (disease)
- 主穴 (main_points)
- 配穴 (auxiliary_points)
- 取穴位置 (position)
- 操作方法 (method)
- 疗程 (course)
- 疗效或经验总结 (effect)

注意：
1. 同一疾病，若主穴+配穴组合相同或高度相似，尽量合并到同一条记录里，避免重复。
2. main_points、auxiliary_points 建议使用数组形式，如 ["太阳", "合谷"]。
3. 若某字段文献未提及，可用空字符串 "" 或空数组 []。
4. 输出标准 JSON 数组格式，如：
   [
     {{
       "disease": "...",
       "main_points": ["..."],
       "auxiliary_points": ["..."],
       "position": "...",
       "method": "...",
       "course": "...",
       "effect": "..."
     }},
     ...
   ]
文献内容如下：
{content}

请直接输出 JSON 数组，不要添加任何解释性文字。
        """.strip()

    def build_prompt_for_acupoint_info(self, content: str) -> str:
        """
        构造用于穴位信息提取的Prompt
        """
        return f"""
你是一位专业的穴位信息整理专员，请对以下文本内容进行结构化提取，并输出为标准 JSON。

需要提取字段：
- 穴位名 (point_name)
- 拼音 (pinyin)
- 国际标准化代号 (standard_code)
- 定位 (location)
- 经络 (meridian)

注意：
1. 每个穴位对应一条记录，确保信息准确对应。
2. 拼音不需要加声调
3. 定位信息需整合正文及注释中的核心描述，去除冗余内容。
4. 经络名称需使用规范名称。
5. 若某字段文本未提及，用空字符串 "" 表示。
6. 输出标准 JSON 数组格式，如：
   [
     {{
       "point_name": "...",
       "pinyin": "...",
       "standard_code": "...",
       "location": "...",
       "meridian": "..."
     }},
     ...
   ]
7.标注文本中的所有内容，不要省略。
文本内容如下：
{content}
阅读全文后
请直接输出 JSON 数组，不要添加任何解释性文字。
        """.strip()

    def build_prompt_for_jin_san_zhen_combo(self, content: str) -> str:
        return f"""
    你是一位专业的靳三针文献整理专员，请对以下文本内容进行结构化提取，并输出为标准 JSON。

    需要提取字段：
    - 穴位组名称 (point_group_name)
    - 穴位组主治 (indications)
    - 穴位 (points)
    - 针刺方法 (acupuncture_method)

    注意：
    1. 每个穴位组对应一条记录，确保信息准确对应。
    2. point_group_name应为具体的靳三针组合名称，如"脑三针"、"醒神针"等
    3. indications为主治症状或疾病
    4. points为该组合包含的具体穴位列表，使用数组形式如["百会","四神针"]
    5. acupuncture_method为每个穴位的针刺操作方法，需要详细描述，格式为字典形式如{{"百会":"斜刺1寸", "四神针":"直刺0.8寸"}}
    6. 若某字段文本未提及，用空字符串""表示，数组字段可用空数组[]
    7. 输出标准 JSON 数组格式，如：
       [
         {{
           "point_group_name": "...",
           "indications": "...",
           "points": ["...", "..."],
           "acupuncture_method": {{...}}
         }},
         ...
       ]
    8. 标注文本中的所有靳三针组合，不要省略。

    文本内容如下：
    {content}

    请仔细阅读全文后，直接输出 JSON 数组，不要添加任何解释性文字。
        """.strip()

    def label_single_txt(self, txt_path: str, prompt_type: str = "acupoint"):
        """
        对单个 TXT 文件进行标注，并输出 jsonl 文件。

        Args:
            txt_path (str): 文本文件路径
            prompt_type (str): 使用哪种提示类型 ("acupoint" 或 "disease")
        """
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        output_file = os.path.join(self.OUT_DIR, f"{base_name}.jsonl")

        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            print(f"⚠️ 文件为空，跳过：{txt_path}")
            return

        # 根据类型选择不同的prompt构建函数
        if prompt_type == "disease":
            prompt = self.build_prompt_for_disease_treatment(content)
        elif prompt_type == "jin_san_zhen_combo":
            prompt = self.build_prompt_for_jin_san_zhen_combo(content)
        else:
            prompt = self.build_prompt_for_acupoint_info(content)

        try:
            response = self.client.chat.completions.create(
                model="qwen3-max",
                messages=[
                    {"role": "system", "content": "你是一位精通中医文献分析的大模型助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
        except Exception as e:
            print(f"❌ 调用模型失败：{txt_path}，错误：{e}")
            return

        result_text = response.choices[0].message.content.strip()

        # 尝试解析 JSON
        try:
            data = json.loads(result_text)
        except json.JSONDecodeError:
            print(f"⚠️ {txt_path} 模型返回内容不是有效 JSON，请手动检查：")
            print(result_text[:1000])
            return

        if not isinstance(data, list):
            print(f"⚠️ {txt_path} 返回 JSON 顶层不是数组，请检查：")
            print(result_text[:1000])
            return

        # 写 jsonl：一条记录一行
        with open(output_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"✅ 标注完成：{txt_path} ，（共 {len(data)} 条）")

    def batch_process(self, prompt_type: str = "acupoint"):
        """
        批量处理所有txt文件

        Args:
            prompt_type (str): 使用哪种提示类型 ("acupoint" 或 "disease")
        """
        txt_files = sorted(glob.glob(os.path.join(self.TXT_DIR, "*.txt")))
        if not txt_files:
            print(f"❌ 未在 {self.TXT_DIR} 中找到 txt 文件，请先运行 extract_text.py")
        else:
            print(f"🔎 共找到 {len(txt_files)} 个 txt，将逐个调用大模型进行标注...")

            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                # 提交所有任务
                futures = [executor.submit(self.label_single_txt, txt, prompt_type) for txt in txt_files]

                # 显示进度条
                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        future.result()  # 获取结果（异常会在这里抛出）
                    except Exception as e:
                        print(f"❌ 处理文件时发生错误: {e}")

            print("🎉 全部 txt 标注流程结束。")


# 主程序入口示例
if __name__ == "__main__":
    # 创建标注器实例
    labeler = BatchAutoLabeler()

    # 处理单个文件示例
    labeler.label_single_txt("extracted_texts/GBT+12346-2021.txt")

    # 批量处理示例（取消注释即可使用）
    # labeler.batch_process(prompt_type="acupoint")
