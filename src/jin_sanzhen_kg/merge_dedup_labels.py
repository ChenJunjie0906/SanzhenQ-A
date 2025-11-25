
import os
import glob
import json


IN_DIR = "labeled_jsonl"

OUT_FILE = "all_marked_merged.jsonl"


def has_valid_points(item):
    """
    检查记录是否包含有效的主穴（只要主穴不为空就保留）
    """
    main_points = item.get("main_points", [])

    # 检查主穴是否为空
    main_empty = (not main_points or
                 (isinstance(main_points, list) and len(main_points) == 0) or
                 (isinstance(main_points, str) and not main_points.strip()))

    # 只要主穴非空就保留
    return not main_empty


def merge_jsonl():
    jsonl_files = sorted(glob.glob(os.path.join(IN_DIR, "*.jsonl")))
    if not jsonl_files:
        print(f"❌ 未在 {IN_DIR} 中找到 jsonl 文件，请先运行 batch_auto_label.py")
        return

    valid_items = []
    total_count = 0
    valid_count = 0

    for path in jsonl_files:
        print(f"📂 处理文件：{path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"⚠️ 跳过非法 JSON 行：{line[:80]}...")
                    continue

                total_count += 1

                # 检查是否包含有效的主穴
                if has_valid_points(item):
                    valid_items.append(item)
                    valid_count += 1
                else:
                    disease = item.get('disease', 'Unknown')
                    main_points = item.get('main_points', [])
                    print(f"🗑️ 过滤掉无效记录：疾病={disease}, 主穴={main_points}")

    # 输出过滤后的 jsonl
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for item in valid_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("========================================")
    print(f"🔢 输入总记录数：{total_count}")
    print(f"✅ 有效记录数：{valid_count}")
    print(f"🗑️ 过滤掉记录数：{total_count - valid_count}")
    print(f"💾 已保存合并结果：{OUT_FILE}")
    print("========================================")


if __name__ == "__main__":
    merge_jsonl()