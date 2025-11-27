import os
import glob
import json


class LabelMerger:
    def __init__(self, input_dir="labeled_jsonl", output_file="all_marked_merged.jsonl"):
        """
        初始化标签合并器

        Args:
            input_dir (str): 包含待合并jsonl文件的目录
            output_file (str): 合并后输出的文件名
        """
        self.input_dir = input_dir
        self.output_file = output_file

    def has_valid_points(self, item):
        """
        检查记录是否包含有效的主穴（只要主穴不为空就保留）

        Args:
            item (dict): 待检查的数据项

        Returns:
            bool: 如果主穴非空返回True，否则返回False
        """
        main_points = item.get("main_points", [])

        # 检查主穴是否为空
        main_empty = (not main_points or
                      (isinstance(main_points, list) and len(main_points) == 0) or
                      (isinstance(main_points, str) and not main_points.strip()))

        # 只要主穴非空就保留
        return not main_empty

    def merge(self):
        """
        执行合并操作，将input_dir中的所有jsonl文件合并为一个文件

        Returns:
            bool: 合并成功返回True，失败返回False
        """
        jsonl_files = sorted(glob.glob(os.path.join(self.input_dir, "*.jsonl")))
        if not jsonl_files:
            print(f"未在 {self.input_dir} 中找到 jsonl 文件，请先运行 batch_auto_label.py")
            return False

        valid_items = []
        total_count = 0
        valid_count = 0

        for path in jsonl_files:
            print(f"处理文件：{path}")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"跳过非法 JSON 行：{line[:80]}...")
                        continue

                    total_count += 1

                    # 检查是否包含有效的主穴
                    if self.has_valid_points(item):
                        valid_items.append(item)
                        valid_count += 1
                    else:
                        disease = item.get('disease', 'Unknown')
                        main_points = item.get('main_points', [])
                        print(f"🗑️ 过滤掉无效记录：疾病={disease}, 主穴={main_points}")

        # 输出过滤后的 jsonl
        with open(self.output_file, "w", encoding="utf-8") as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print("========================================")
        print(f"输入总记录数：{total_count}")
        print(f"有效记录数：{valid_count}")
        print(f"过滤掉记录数：{total_count - valid_count}")
        print(f"已保存合并结果：{self.output_file}")
        print("========================================")

        return True


if __name__ == "__main__":
    merger = LabelMerger()
    merger.merge()

