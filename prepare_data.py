# =============================================================================
# 1. 匯入套件
# =============================================================================
import json
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

# =============================================================================
# 2. 資料設定
# =============================================================================

# 原始資料檔。
INPUT_FILE = ("hf://datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset/data.csv")
# 輸出資料夾。
OUTPUT_DIR = Path("data/bert")

# Validation 比例。
VALID_RATIO = 0.2
SEED = 42

EMOTIONS = [
    "平淡語氣",
    "開心語調",
    "關切語調",
    "憤怒語調",
    "驚奇語調",
    "悲傷語調",
    "厭惡語調",
    "疑問語調",
]

# label 0-7
LABELID = {emotion: i for i, emotion in enumerate(EMOTIONS)}

# =============================================================================
# 3. 輔助函式
# =============================================================================

# 讀取 CSV、清理資料、轉換 Label
def load_reviews(path):

    df = pd.read_csv(path)

    # 清理缺失值
    df = df.dropna(subset=["text", "emotion"]).copy()

    # 去除前後空白
    df["text"] = df["text"].str.strip()
    df["emotion"] = df["emotion"].str.strip()

    # 排除空文字
    df = df[df["text"] != ""]

    # 檢查未知的情緒
    unknown = set(df["emotion"]) - set(LABELID)

    if unknown:
        raise ValueError(f"未知情緒：{unknown}")

    # 文字 Label 轉換成整數
    df["labels"] = df["emotion"].map(LABELID).astype(int)

    texts = df["text"].tolist()
    labels = df["labels"].tolist()

    if not texts:
        raise RuntimeError("沒有有效資料")

    return texts, labels

# 將處理好的資料儲存成 JSONL
def save_jsonl(texts, labels, path: Path):
    with path.open("w", encoding="utf-8") as file:
        for text, label in zip(texts, labels):

            row = {
                "sentences": text,
                "labels": label,
            }

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

# 顯示各情緒的資料筆數及比例
def print_label_distribution(name, labels):

    counts = Counter(labels)
    total = len(labels)

    print(f"{name}：{total} 筆")

    for label in sorted(counts):
        count = counts[label]
        ratio = count / total

        print(
            f"  label={label}: "
            f"{count} 筆 "
            f"({ratio:.2%})"
        )


# =============================================================================
# 4. 主流程
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Step 1：讀取 reviews.txt
    # -------------------------------------------------------------------------

    texts, labels = load_reviews(INPUT_FILE)

    print("=" * 72)
    print_label_distribution("全部資料", labels)

    # -------------------------------------------------------------------------
    # Step 2：切 train / validation
    # -------------------------------------------------------------------------
    (
        train_texts,
        valid_texts,
        train_labels,
        valid_labels,
    ) = train_test_split(
        texts,
        labels,
        test_size=VALID_RATIO,
        random_state=SEED,
        stratify=labels,
    )

    # -------------------------------------------------------------------------
    # Step 3：建立輸出資料夾
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = OUTPUT_DIR / "train.jsonl"
    valid_path = OUTPUT_DIR / "valid.jsonl"

    # -------------------------------------------------------------------------
    # Step 4：儲存 JSONL
    # -------------------------------------------------------------------------

    save_jsonl(
        train_texts,
        train_labels,
        train_path,
    )

    save_jsonl(
        valid_texts,
        valid_labels,
        valid_path,
    )

    # -------------------------------------------------------------------------
    # Step 5：顯示結果
    # -------------------------------------------------------------------------

    print_label_distribution(
        "Training",
        train_labels,
    )

    print_label_distribution(
        "Validation",
        valid_labels,
    )

    print(f"Training JSONL：{train_path}")
    print(f"Validation JSONL：{valid_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
