import os
import json
import torch
import gc
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import KFold
from peft import LoraConfig, get_peft_model

# 載入 JSON 檔案
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 轉換為 Hugging Face Datasets 格式
def prepare_dataset(data):
    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]
    return Dataset.from_dict({"question": questions, "answer": answers})

# 合併問題與答案，並調整序列長度（max_length 設為 64）
def preprocess_function(examples, tokenizer, max_length=64):
    # 輸入格式：Q: 問題\nA:
    inputs = ["Q: " + q.strip() + "\nA:" for q in examples['question']]
    targets = [a.strip() for a in examples['answer']]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=max_length, truncation=True, padding=True)["input_ids"]
    # 調整 labels 長度，若不足則用 -100 填充（loss 時忽略 -100）
    model_inputs["labels"] = [
        label + [-100] * (len(model_inputs["input_ids"][i]) - len(label))
        if len(label) < len(model_inputs["input_ids"][i])
        else label[:len(model_inputs["input_ids"][i])]
        for i, label in enumerate(labels)
    ]
    return model_inputs

def main():
    # 設定 CUDA 配置與部分優化參數
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
    torch.backends.cuda.matmul.allow_tf32 = True

    # 檢查是否有可用的 CUDA 裝置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置: {device}")

    # 載入資料並轉換為 dataset
    file_path = r"C:\Users\ai\Desktop\春日部\fine tune\金管會問題集 JSON格式.json"
    raw_data = load_data(file_path)
    dataset = prepare_dataset(raw_data)

    # 初始化分詞器與模型名稱
    model_name = "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 預處理資料，設定 max_length 為 64
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer, max_length=64), batched=True)

    # LoRA 配置
    lora_config = LoraConfig(
        r=8,                   # LoRA 矩陣的秩（rank）
        lora_alpha=16,         # 調整 scaling factor
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 使用 K 折交叉驗證，設定 3 折
    k = 3
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_indices, eval_indices) in enumerate(kfold.split(tokenized_dataset)):
        torch.cuda.empty_cache()  # 清理顯存
        print(f"\n=== 開始第 {fold + 1}/{k} 折訓練 ===")

        train_dataset = tokenized_dataset.select(train_indices)
        eval_dataset = tokenized_dataset.select(eval_indices)

        training_args = TrainingArguments(
            output_dir=f"./results_fold_{fold + 1}",
            overwrite_output_dir=True,
            num_train_epochs=5,                   # 訓練 5 個 epoch
            per_device_train_batch_size=1,
            gradient_accumulation_steps=32,       # 降低梯度累積步數
            save_steps=1000,
            save_total_limit=1,
            logging_dir=f"./logs_fold_{fold + 1}",
            logging_steps=50,
            eval_strategy="epoch",               # 每個 epoch 結束後進行評估
            learning_rate=5e-4,
            fp16=True,
            report_to="none"
        )

        # 初始化模型並啟用 gradient checkpointing
        base_model = LlamaForCausalLM.from_pretrained(model_name).to(device)
        base_model.gradient_checkpointing_enable()
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        # 開始訓練與評估
        trainer.train()
        eval_result = trainer.evaluate()
        print(f"第 {fold + 1} 折評估結果: {eval_result}")
        fold_results.append(eval_result)

        # 清除資源釋放顯存
        del trainer
        del model
        torch.cuda.empty_cache()

    # 使用完整資料集進行最終微調
    print("\n=== 完成 K 折，進行最終微調 ===")
    training_args = TrainingArguments(
        output_dir="./final_model",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        save_steps=1000,
        save_total_limit=1,
        logging_dir="./logs_final",
        logging_steps=50,
        eval_strategy="no",  # 最終微調時不進行評估
        learning_rate=5e-4,
        fp16=True,
        report_to="none"
    )
    base_model = LlamaForCausalLM.from_pretrained(model_name).to(device)
    base_model.gradient_checkpointing_enable()
    model = get_peft_model(base_model, lora_config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    save_path = r"C:\Users\ai\Desktop\春日部\fine tune\微調後的模型\TAIDE-4\final_model"
    trainer.save_model(save_path)
    print(f"最終模型已保存至 {save_path}")

if __name__ == "__main__":
    main()
