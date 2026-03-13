import json
import random
import os

def sample_with_replacement(items, count, rng):
    if count <= 0:
        return []
    if not items:
        return []
    return [rng.choice(items) for _ in range(count)]

def sample_without_replacement(items, count, rng):
    if count <= 0:
        return []
    if count >= len(items):
        copied = items[:]
        rng.shuffle(copied)
        return copied
    return rng.sample(items, count)

def main():
    input_file = os.getenv("INPUT_FILE", "data/processed/labeled_samples.json")
    output_file = os.getenv("OUTPUT_FILE", "data/processed/labeled_samples_balanced.json")
    target_positive_ratio = float(os.getenv("TARGET_POSITIVE_RATIO", "0.30"))
    total_scale = float(os.getenv("TOTAL_SCALE", "1.0"))
    seed = int(os.getenv("RANDOM_SEED", "42"))

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"未找到输入文件: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("输入数据为空，无法重采样")

    rng = random.Random(seed)
    negatives = [x for x in data if x.get("label") == 0]
    positives = [x for x in data if x.get("label") == 1]
    if not negatives or not positives:
        raise ValueError("标签分布异常，必须同时包含 label=0 和 label=1")

    original_total = len(data)
    target_total = max(2, int(round(original_total * total_scale)))
    target_positive = min(target_total - 1, max(1, int(round(target_total * target_positive_ratio))))
    target_negative = target_total - target_positive

    sampled_negative = sample_without_replacement(negatives, target_negative, rng)
    sampled_positive = sample_with_replacement(positives, target_positive, rng)

    resampled = sampled_negative + sampled_positive
    rng.shuffle(resampled)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(resampled, f, ensure_ascii=False, indent=2)

    new_neg = sum(1 for x in resampled if x.get("label") == 0)
    new_pos = sum(1 for x in resampled if x.get("label") == 1)
    print(f"原始样本总数: {original_total}")
    print(f"原始分布: label=0 {len(negatives)} | label=1 {len(positives)}")
    print(f"目标总数: {target_total}")
    print(f"新分布: label=0 {new_neg} | label=1 {new_pos}")
    print(f"新正样本占比: {new_pos / len(resampled):.4f}")
    print(f"重采样完成，已保存到: {output_file}")

if __name__ == "__main__":
    main()
