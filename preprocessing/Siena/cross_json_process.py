import os
import sys
import json
import pickle
import random
import re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np


data_root = sys.argv[1]
print(f"Data root: {data_root}")

processed_data_path = Path(data_root) / "Siena" / "processed_data"
data_split_path = Path("./preprocessing/Siena/cross_subject_json")
data_split_path.mkdir(parents=True, exist_ok=True)

save_train_path = data_split_path / "train.json"
save_val_path = data_split_path / "val.json"
save_test_path = data_split_path / "test.json"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Siena 预处理一般是 10 秒片段；如果你后面检查出来不是 10，再改这里
SEGMENT_SECONDS = 10

# 根据你日志里的 Siena 通道顺序，29 通道
SIENA_CH_NAMES_29 = [
    "Fp1", "F3", "C3", "P3", "O1",
    "F7", "T3", "T5", "Fc1", "Fc5",
    "Cp1", "Cp5", "F9", "Fz", "Cz", "Pz",
    "FP2", "F4", "C4", "P4", "O2",
    "F8", "T4", "T6", "Fc2", "Fc6",
    "Cp2", "Cp6", "F10"
]


def get_subject_name(pkl_path: Path) -> str:
    """
    优先从路径中抓 PNxx，例如：
    dataset/Siena/processed_data/PN12/xxx.pkl
    或者文件名里带 PN12。
    """
    text = str(pkl_path)
    m = re.search(r"(PN\d+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 兜底：用 processed_data 后第一层目录名
    try:
        rel = pkl_path.relative_to(processed_data_path)
        if len(rel.parts) > 1:
            return rel.parts[0]
    except Exception:
        pass

    # 再兜底：用文件名前缀
    return pkl_path.stem.split("_")[0].split("-")[0]


def load_sample(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if "X" not in obj:
        raise KeyError(f"{pkl_path} does not contain key 'X'")

    X = np.asarray(obj["X"])

    if "Y" in obj:
        y = obj["Y"]
    elif "label" in obj:
        y = obj["label"]
    else:
        raise KeyError(f"{pkl_path} does not contain key 'Y' or 'label'")

    # label 转成 int，兼容 numpy/list/scalar
    if isinstance(y, (list, tuple, np.ndarray)):
        y = np.asarray(y).reshape(-1)[0]
    y = int(y)

    return X, y


def build_items(pkl_files):
    subject_id_map = {}
    subject_id_counter = 0
    items = []
    label_counter = Counter()

    first_shape = None

    for p in pkl_files:
        try:
            X, y = load_sample(p)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")
            continue

        if X.ndim != 2:
            print(f"[WARN] skip {p}: X shape is {X.shape}, expected 2D (channels, time)")
            continue

        subject_name = get_subject_name(p)
        if subject_name not in subject_id_map:
            subject_id_map[subject_name] = subject_id_counter
            subject_id_counter += 1

        if first_shape is None:
            first_shape = X.shape

        items.append({
            "subject_id": subject_id_map[subject_name],
            "subject_name": subject_name,
            "file": str(p),
            "label": y
        })
        label_counter[y] += 1

    return items, subject_id_map, label_counter, first_shape


def compute_stats(items):
    if len(items) == 0:
        raise RuntimeError("No items available for computing normalization stats.")

    first_X, _ = load_sample(Path(items[0]["file"]))
    num_channels = first_X.shape[0]

    total_mean = np.zeros(num_channels, dtype=np.float64)
    total_std = np.zeros(num_channels, dtype=np.float64)
    max_value = -np.inf
    min_value = np.inf
    n = 0

    for item in items:
        X, _ = load_sample(Path(item["file"]))
        if X.shape[0] != num_channels:
            print(f"[WARN] channel mismatch, skip stats: {item['file']} shape={X.shape}")
            continue

        total_mean += X.mean(axis=-1)
        total_std += X.std(axis=-1)
        max_value = max(max_value, float(X.max()))
        min_value = min(min_value, float(X.min()))
        n += 1

    if n == 0:
        raise RuntimeError("No valid samples for stats.")

    return {
        "min": min_value,
        "max": max_value,
        "mean": (total_mean / n).tolist(),
        "std": (total_std / n).tolist()
    }


def save_json(items, save_path, dataset_info):
    data = {
        "dataset_info": dataset_info,
        "subject_data": items
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {save_path}, samples={len(items)}")


def main():
    pkl_files = sorted(processed_data_path.rglob("*.pkl"))

    print("processed_data_path:", processed_data_path)
    print("total pkl files:", len(pkl_files))

    if len(pkl_files) == 0:
        raise RuntimeError(
            "No pkl files found. Check whether data_process.py saved files under dataset/Siena/processed_data."
        )

    all_items, subject_id_map, label_counter, first_shape = build_items(pkl_files)

    print("valid samples:", len(all_items))
    print("subjects:", len(subject_id_map), sorted(subject_id_map.keys()))
    print("label distribution:", dict(label_counter))
    print("first X shape:", first_shape)

    if len(all_items) == 0:
        raise RuntimeError("No valid samples loaded from pkl files.")

    # 按 subject 切分，避免同一个病人的片段同时出现在 train/test
    subjects = sorted(set(item["subject_name"] for item in all_items))
    random.shuffle(subjects)

    n_sub = len(subjects)
    n_train = max(1, int(n_sub * 0.7))
    n_val = max(1, int(n_sub * 0.15))

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train + n_val])
    test_subjects = set(subjects[n_train + n_val:])

    # 如果 subject 太少导致 test 为空，就从 train 里挪一个
    if len(test_subjects) == 0 and len(train_subjects) > 1:
        moved = sorted(train_subjects)[-1]
        train_subjects.remove(moved)
        test_subjects.add(moved)

    train_items = [x for x in all_items if x["subject_name"] in train_subjects]
    val_items = [x for x in all_items if x["subject_name"] in val_subjects]
    test_items = [x for x in all_items if x["subject_name"] in test_subjects]

    print("train subjects:", sorted(train_subjects))
    print("val subjects:", sorted(val_subjects))
    print("test subjects:", sorted(test_subjects))
    print("split sizes:", len(train_items), len(val_items), len(test_items))

    if len(train_items) == 0:
        raise RuntimeError("Train split is empty.")
    if len(val_items) == 0:
        print("[WARN] Val split is empty.")
    if len(test_items) == 0:
        print("[WARN] Test split is empty.")

    stats = compute_stats(train_items)

    # 从第一个样本估计采样率。Siena 日志看起来原始多为 512Hz。
    first_X, _ = load_sample(Path(train_items[0]["file"]))
    num_channels, num_points = first_X.shape
    estimated_sampling_rate = int(round(num_points / SEGMENT_SECONDS))

    if num_channels == 29:
        ch_names = SIENA_CH_NAMES_29
    else:
        ch_names = [f"CH{i}" for i in range(num_channels)]

    dataset_info = {
        "sampling_rate": estimated_sampling_rate,
        "ch_names": ch_names,
        "min": stats["min"],
        "max": stats["max"],
        "mean": stats["mean"],
        "std": stats["std"]
    }

    print("dataset_info sampling_rate:", estimated_sampling_rate)
    print("dataset_info num_channels:", num_channels)
    print("num_t should be:", num_points / estimated_sampling_rate)

    save_json(train_items, save_train_path, dataset_info)
    save_json(val_items, save_val_path, dataset_info)
    save_json(test_items, save_test_path, dataset_info)


if __name__ == "__main__":
    main()