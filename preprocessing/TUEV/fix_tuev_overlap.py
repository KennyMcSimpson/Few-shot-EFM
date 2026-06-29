from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "dataset" / "TUEV" / "processed_data" / "train_dir"
EVAL_DIR = ROOT / "dataset" / "TUEV" / "processed_data" / "eval_dir"
REPORT_PATH = ROOT / "tuev_overlap_report.txt"

CHECK_HASH = True


def file_hash(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train folder not found: {TRAIN_DIR}")
    if not EVAL_DIR.exists():
        raise FileNotFoundError(f"Eval folder not found: {EVAL_DIR}")

    train_files = {p.name: p for p in TRAIN_DIR.glob("*.pkl")}
    eval_files = {p.name: p for p in EVAL_DIR.glob("*.pkl")}

    overlap_names = sorted(set(train_files.keys()) & set(eval_files.keys()))

    same_size = []
    diff_size = []
    same_hash = []
    diff_hash = []

    for name in overlap_names:
        train_path = train_files[name]
        eval_path = eval_files[name]

        train_size = train_path.stat().st_size
        eval_size = eval_path.stat().st_size

        if train_size == eval_size:
            same_size.append(name)
        else:
            diff_size.append((name, train_size, eval_size))

        if CHECK_HASH:
            train_hash = file_hash(train_path)
            eval_hash = file_hash(eval_path)
            if train_hash == eval_hash:
                same_hash.append(name)
            else:
                diff_hash.append(name)

    lines = []
    lines.append("TUEV train/eval overlap check")
    lines.append("=" * 60)
    lines.append(f"train_dir: {TRAIN_DIR}")
    lines.append(f"eval_dir : {EVAL_DIR}")
    lines.append("")
    lines.append(f"train .pkl count: {len(train_files)}")
    lines.append(f"eval  .pkl count: {len(eval_files)}")
    lines.append(f"same filename count: {len(overlap_names)}")
    lines.append(f"same filename + same size count: {len(same_size)}")

    if CHECK_HASH:
        lines.append(f"same filename + same hash count: {len(same_hash)}")
        lines.append(f"same filename but different hash count: {len(diff_hash)}")

    lines.append("")
    lines.append("First 50 overlapping filenames:")
    for name in overlap_names[:50]:
        lines.append(name)

    if diff_size:
        lines.append("")
        lines.append("Files with same filename but different size:")
        for name, train_size, eval_size in diff_size[:50]:
            lines.append(f"{name} | train_size={train_size} | eval_size={eval_size}")

    if diff_hash:
        lines.append("")
        lines.append("Files with same filename but different hash:")
        for name in diff_hash[:50]:
            lines.append(name)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 60)
    print("TUEV train/eval overlap check")
    print("=" * 60)
    print(f"train .pkl count: {len(train_files)}")
    print(f"eval  .pkl count: {len(eval_files)}")
    print(f"same filename count: {len(overlap_names)}")
    print(f"same filename + same size count: {len(same_size)}")

    if CHECK_HASH:
        print(f"same filename + same hash count: {len(same_hash)}")
        print(f"same filename but different hash count: {len(diff_hash)}")

    print(f"Report saved to: {REPORT_PATH}")

    if len(overlap_names) > 0:
        print("")
        print("Conclusion: train_dir and eval_dir DO overlap.")
        if CHECK_HASH and len(same_hash) > 0:
            print("The overlapping files are not just same-name; at least some are exactly identical by hash.")
    else:
        print("")
        print("Conclusion: no same-name overlap between train_dir and eval_dir.")


if __name__ == "__main__":
    main()
