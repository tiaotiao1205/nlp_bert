# =============================================================================
# 1. 匯入套件
# =============================================================================

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers

from datasets import load_dataset

from sklearn.metrics import f1_score

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# =============================================================================
# 2. 模型與路徑設定
# =============================================================================

# Hugging Face 預訓練模型。
MODEL_NAME = "google-bert/bert-base-chinese"

# prepare_data.py 的輸出資料夾。
DATA_DIR = Path("data/bert")

TRAIN_FILE = "train.jsonl"
VALID_FILE = "valid.jsonl"

# Fine-tuned model 與 checkpoints 的輸出位置。
OUTPUT_DIR = Path("output")

# =============================================================================
# 3. 模型設定
# =============================================================================

NUM_LABELS = 8

# 數字轉情緒名稱
ID2LABEL = {
    0: "平淡語氣",
    1: "開心語調",
    2: "關切語調",
    3: "憤怒語調",
    4: "驚奇語調",
    5: "悲傷語調",
    6: "厭惡語調",
    7: "疑問語調",
}

# 情緒名稱轉數字
LABEL2ID = {
    emotion: idx
    for idx, emotion in ID2LABEL.items()
}
# BERT-base-chinese 的最大 context window 是 512 tokens。
MAX_SEQ_LENGTH = 128

# =============================================================================
# 4. 訓練設定
# =============================================================================
SEED = 42
NUM_TRAIN_EPOCHS = 3
# 一次 forward/backward 真正送進 CPU 的 examples 數。
PER_DEVICE_TRAIN_BATCH_SIZE = 8
# Validation 不需要 backward，
# 通常可以使用相同或更大的 batch。
PER_DEVICE_EVAL_BATCH_SIZE = 8
# 因此大約每約 16 筆 training examples 更新一次（8 × 2）模型參數。
GRADIENT_ACCUMULATION_STEPS = 2
# 1e-4 是一個獨立選擇的 learning-rate hyperparameter。
LEARNING_RATE = 1e-4
# 前 50 個 optimizer steps 做 linear warmup。
WARMUP_STEPS = 50
# 權重衰減 (weight decay)，主要是用在防止過度擬合（overfitting）。
WEIGHT_DECAY = 0.01
# 學習率調整策略 (learning rate scheduler)。
LR_SCHEDULER_TYPE = "linear"
# 因此這裡改成每 5 個 optimizer steps 做一次 validation，
# 整個 30 epochs 大約會有 9～10 次 validation。
EVAL_STEPS = 30
# 如此 load_best_model_at_end=True 才能穩定追蹤最佳 checkpoint。
SAVE_STEPS = 30
# 最多保留 2 個 checkpoints。
#
# 開啟 load_best_model_at_end=True 後，
# Trainer 會盡量保留最佳 checkpoint，
# 同時限制其他 checkpoints 不要無限制佔用磁碟。
SAVE_TOTAL_LIMIT = 2
# 這時就會觸發 early stopping，
# 不一定需要把 NUM_TRAIN_EPOCHS=3 全部跑完。
EARLY_STOPPING_PATIENCE = 3
# 每 10 個 optimizer steps 印一次 training log。
LOGGING_STEPS = 10

# =============================================================================
# 5. Mixed Precision
# =============================================================================

# Tesla T4 適合 FP16。
#
# 如果不是 CUDA GPU，下面 main() 會自動關閉 fp16。
USE_FP16 = False

# T4 不使用 BF16。
USE_BF16 = False


# =============================================================================
# 6. 輔助函式
# =============================================================================
def print_gpu_info():
    """
    顯示目前 PyTorch 是否看到 GPU。
    """

    print("=" * 72)
    print(f"PyTorch：{torch.__version__}")
    print(f"Transformers：{transformers.__version__}")
    print(f"CUDA available：{torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(
            "CUDA capability："
            f"{torch.cuda.get_device_capability(0)}"
        )
        print(f"PyTorch CUDA：{torch.version.cuda}")

    print("=" * 72)

def compute_metrics(eval_pred):

    labels = eval_pred.label_ids

    predictions = eval_pred.predictions

    # 某些模型/Trainer 設定可能回傳 tuple，
    # 這裡保留第一個 logits tensor。
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = np.argmax(
        predictions,
        axis=-1,
    )

    f1 = f1_score(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    return {
        "f1": f1,
    }

# =============================================================================
# 7. 主流程
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Step 1：GPU 資訊
    # -------------------------------------------------------------------------

    print_gpu_info()

    # -------------------------------------------------------------------------
    # Step 2：確認資料
    # -------------------------------------------------------------------------

    train_path = DATA_DIR / TRAIN_FILE
    valid_path = DATA_DIR / VALID_FILE

    if not train_path.is_file():
        raise FileNotFoundError(
            f"找不到 training data：{train_path.resolve()}\n"
            "請先執行 python prepare_data.py"
        )

    if not valid_path.is_file():
        raise FileNotFoundError(
            f"找不到 validation data：{valid_path.resolve()}\n"
            "請先執行 python prepare_data.py"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Step 3：讀取 JSONL
    # -------------------------------------------------------------------------

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(valid_path),
        },
    )

    print(dataset)

    # -------------------------------------------------------------------------
    # Step 4：載入 tokenizer
    # -------------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # -------------------------------------------------------------------------
    # Step 5：Tokenization
    # -------------------------------------------------------------------------

    def tokenize_batch(batch):
        """
        只做 tokenization + truncation。

        注意這裡刻意不使用：

            padding=True

        因為我們會在真正組成 mini-batch 時，
        使用 DataCollatorWithPadding 做 dynamic padding。

        這樣每個 batch 只補到該 batch 最長序列，
        而不是預先補很多不必要的 PAD tokens。
        """

        return tokenizer(
            batch["sentences"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,

        # 原始文字不需要在 model.forward() 時傳進去，
        # tokenization 後可以移除 sentences 欄位。
        remove_columns=["sentences"],
    )

    # -------------------------------------------------------------------------
    # Step 6：Dynamic Padding
    # -------------------------------------------------------------------------

    # 假設同一個 mini-batch 三筆長度：
    #
    #     80 tokens
    #     130 tokens
    #     200 tokens
    #
    # DataCollatorWithPadding 只會補到：
    #
    #     200 tokens
    #
    # 而不是全部補到 MAX_SEQ_LENGTH=128。
    #
    # 對短文本資料通常可以省下不少計算。
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    # -------------------------------------------------------------------------
    # Step 7：建立 BERT sequence classifier
    # -------------------------------------------------------------------------

    # bert-base-chinese 原本只有 pretrained encoder。
    #
    # AutoModelForSequenceClassification 會在 BERT 上加一個
    # sequence-classification head。
    #
    # 因為 NUM_LABELS=8，
    # 最後 logits shape 會是：
    #
    #     [batch_size, 2]
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # -------------------------------------------------------------------------
    # Step 8：TrainingArguments
    # -------------------------------------------------------------------------

    # 只有真的有 CUDA 時才啟用 FP16。
    fp16_enabled = (
        USE_FP16
        and torch.cuda.is_available()
    )

    training_args = TrainingArguments(
        # Trainer 的 checkpoint 與最後輸出位置。
        #
        # 注意：
        # 舊版 Transformers 曾支援：
        #
        #     overwrite_output_dir=True
        #
        # 但目前 Colab 安裝的新版 TrainingArguments 已不接受
        # overwrite_output_dir 這個參數，因此這裡不再傳入。
        #
        # 如果你想完全重新訓練，最乾淨的方法是先自行刪除：
        #
        #     rm -rf output/bert_binary_clf
        #
        # 再執行本程式。
        output_dir=str(OUTPUT_DIR),

        # max_steps=10先試跑 10 個 Training Steps
        num_train_epochs=NUM_TRAIN_EPOCHS,
        #max_steps=10,

        per_device_train_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            PER_DEVICE_EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        warmup_steps=WARMUP_STEPS,

        weight_decay=WEIGHT_DECAY,

        lr_scheduler_type=LR_SCHEDULER_TYPE,

        # 每固定 optimizer steps 做 validation。
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,

        # Evaluation 與 save 使用相同 strategy，
        # 讓 load_best_model_at_end 可以正常追蹤 checkpoint。
        save_strategy="steps",
        save_steps=SAVE_STEPS,

        save_total_limit=SAVE_TOTAL_LIMIT,

        logging_steps=LOGGING_STEPS,

        # 訓練結束後，自動把最佳 checkpoint 載回 trainer.model。
        #
        # 「訓練結束」可能有兩種情況：
        #
        # 1. 正常跑完 NUM_TRAIN_EPOCHS。
        # 2. EarlyStoppingCallback 提前停止。
        #
        # 無論是哪一種，
        # Trainer 最後都會把 validation 指標最佳的 checkpoint
        # 載回 trainer.model。
        load_best_model_at_end=True,

        # 用 validation loss 判斷最佳模型。
        #
        # EarlyStoppingCallback 也會依照這個指標判斷
        # 是否已經連續多次沒有改善。
        metric_for_best_model="eval_loss",

        # eval_loss 越低越好。
        #
        # 因此：
        #
        #     新 eval_loss < best eval_loss
        #
        # 才算 improvement。
        greater_is_better=False,

        fp16=fp16_enabled,
        bf16=USE_BF16,
        tf32=False,

        seed=SEED,
        data_seed=SEED,

        report_to="none",
    )

    # -------------------------------------------------------------------------
    # Step 9：建立 Trainer
    # -------------------------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=(
            tokenized_dataset["train"]
        ),

        eval_dataset=(
            tokenized_dataset["validation"]
        ),

        data_collator=data_collator,

        compute_metrics=compute_metrics,

        # -------------------------------------------------------------
        # Early Stopping
        # -------------------------------------------------------------
        #
        # EarlyStoppingCallback 會讀取：
        #
        #     metric_for_best_model
        #
        # 目前設定是：
        #
        #     eval_loss
        #
        # 所以每次 evaluation 後，
        # callback 會比較新的 eval_loss
        # 和目前最佳 eval_loss。
        #
        # 如果連續 EARLY_STOPPING_PATIENCE 次
        # 都沒有改善，
        # Trainer 就會提早結束。
        #
        # 注意：
        # Early stopping 不代表「最後一個 checkpoint」最好。
        #
        # 因為我們同時設定：
        #
        #     load_best_model_at_end=True
        #
        # 所以 training 結束後，
        # Trainer 仍會把最佳 checkpoint
        # 重新載回 trainer.model。
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=(
                    EARLY_STOPPING_PATIENCE
                )
            )
        ],
    )

    # -------------------------------------------------------------------------
    # Step 10：開始 fine-tuning
    # -------------------------------------------------------------------------

    trainer.train()

    # -------------------------------------------------------------------------
    # Step 11：顯示最佳 checkpoint
    # -------------------------------------------------------------------------

    print(
        "最佳 checkpoint："
        f"{trainer.state.best_model_checkpoint}"
    )

    print(
        "最佳 eval_loss："
        f"{trainer.state.best_metric}"
    )

    # -------------------------------------------------------------------------
    # Step 12：儲存最佳模型
    # -------------------------------------------------------------------------

    # 因為 load_best_model_at_end=True，
    # 此時 trainer.model 已經是最佳 checkpoint。
    trainer.save_model(
        str(OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )

    # trainer_state.json 包含：
    #
    # - log_history
    # - global_step
    # - best_metric
    # - best_model_checkpoint
    #
    # plot_bert_metrics.py 會使用它。
    trainer.save_state()

    print(
        "Final model directory："
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()