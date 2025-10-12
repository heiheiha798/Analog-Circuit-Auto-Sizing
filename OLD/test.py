# 文件路径
file_path = "e:/BaiduSyncdisk/大三上/芯片设计自动化与智能优化-lyb/挑战赛/results_4.2_2.txt"

# 用集合存储所有出现的序号
indices = set()

with open(file_path, encoding="utf-8") as f:
    for line in f:
        if line.startswith("./iter_") and "/params.txt" in line:
            # 提取序号
            try:
                idx = int(line.split("/")[1].split("_")[1].split("/")[0])
                indices.add(idx)
            except Exception:
                pass

# 检查1到518是否都存在
missing = [i for i in range(1, 519) if i not in indices]

if not missing:
    print("1到518的序号全部存在")
else:
    print(f"缺少序号: {missing}")