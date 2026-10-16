from pathlib import Path
from pprint import pprint

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


# =============================================================================
# 1. 推論設定
# =============================================================================

MODEL_DIR = Path("output")

# 可以直接修改這個 list 測試不同句子。
TEST_TEXTS = [
    "我每天都能跟她一起上學，我好開心！",
    "最好的朋友要離開臺灣了，以後可能不容易再見面...",
    "我覺得我快不行了...",
    "剛剛收到研究所錄取的通知書！",
    "今年的冬天好像比較晚來。"
]


# =============================================================================
# 2. 主流程
# =============================================================================

def main():

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到 fine-tuned model：{MODEL_DIR.resolve()}\n"
            "請先執行 train_bert.py。"
        )

    # -------------------------------------------------------------------------
    # Step 1：載入 tokenizer
    # -------------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    # -------------------------------------------------------------------------
    # Step 2：載入 sequence classifier
    # -------------------------------------------------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_DIR
        )
    )

    # -------------------------------------------------------------------------
    # Step 3：決定 inference device
    # -------------------------------------------------------------------------
    device = (
        0
        if torch.cuda.is_available()
        else -1
    )

    print(
        "Inference device："
        + (
            torch.cuda.get_device_name(0)
            if device == 0
            else "CPU"
        )
    )

    # -------------------------------------------------------------------------
    # Step 4：建立 Hugging Face pipeline
    # -------------------------------------------------------------------------

    classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Step 5：執行分類
    # -------------------------------------------------------------------------

    results = classifier(
        TEST_TEXTS,
        truncation=True,
        max_length=512,
    )

    # -------------------------------------------------------------------------
    # Step 6：顯示結果
    # -------------------------------------------------------------------------

    for text, result in zip(
        TEST_TEXTS,
        results,
    ):
        print("=" * 72)
        print(f"文字：{text}")
        print(
            f"預測：{result['label']}"
        )
        print(
            f"信心分數：{result['score']:.4f}"
        )

    print("=" * 72)

    # 如果想直接查看原始 list，也可以保留 pprint。
    print("Raw pipeline output：")
    pprint(results)


if __name__ == "__main__":
    main()
