import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import KFold
from peft import LoraConfig, get_peft_model
import gc

# 設定 Hugging Face 權杖
HF_TOKEN = "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"
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

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

# 合併問題與答案，並進行 tokenization（使用固定長度以降低運算負擔）
def preprocess_function(examples, tokenizer, max_length=48):
    inputs = ["Q: " + q + "\nA:" for q in examples['question']]
    targets = [a for a in examples['answer']]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length")["input_ids"]
    # 將 padding token 的 label 設為 -100，避免計算損失時納入 padding 部分
    model_inputs["labels"] = [
        label + [-100] * (len(model_inputs["input_ids"][i]) - len(label))
        if len(label) < len(model_inputs["input_ids"][i])
        else label[:len(model_inputs["input_ids"][i])]
        for i, label in enumerate(labels)
    ]
    return model_inputs

def main():
    # 設定 CUDA 動態分配與啟用 TensorFloat32
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
    torch.backends.cuda.matmul.allow_tf32 = True

    # 檢查 CUDA 是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 載入資料（請確認檔案為 JSON 格式）
    file_path = r"C:\Users\ai\Desktop\春日部\fine tune\金管會問題集 JSON格式.json"
    raw_data = load_data(file_path)
    dataset = prepare_dataset(raw_data)

    # 初始化分詞器
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # LoRA 配置
    lora_config = LoraConfig(
        r=8,               # 降低 LoRA 矩陣的秩（rank）
        lora_alpha=16,     # 調整 scaling factor
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # K 折交叉驗證設定
    k = 3
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_indices, eval_indices) in enumerate(kfold.split(tokenized_dataset)):
        torch.cuda.empty_cache()  # 清理顯存
        print(f"\n=== 開始第 {fold + 1}/{k} 折 ===")
        
        train_dataset = tokenized_dataset.select(train_indices)
        eval_dataset = tokenized_dataset.select(eval_indices)

        # 訓練參數設定
        training_args = TrainingArguments(
            output_dir=f"./results_fold_{fold + 1}",
            overwrite_output_dir=True,
            num_train_epochs=5,                # 根據需求調整 epoch 數
            per_device_train_batch_size=1,
            gradient_accumulation_steps=96,
            save_steps=1000,
            save_total_limit=1,
            logging_dir=f"./logs_fold_{fold + 1}",
            logging_steps=50,
            evaluation_strategy="epoch",
            learning_rate=5e-4,
            fp16=True,
            report_to="none",
            dataloader_num_workers=0         # Windows 平台建議設為 0
        )

        # 使用 AutoModelForCausalLM 載入模型，並設定 use_cache=False 以搭配 gradient checkpointing
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        base_model.config.use_cache = False
        base_model.to(device)
        base_model.gradient_checkpointing_enable()
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

        # 訓練器設定
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

        # 釋放該折訓練使用的 GPU 記憶體
        del trainer, model, base_model
        torch.cuda.empty_cache()
        gc.collect()

    # 完整資料集進行最終微調
    print("\n=== 完成 K 折驗證，進行最終微調 ===")
    # 調整 gradient_accumulation_steps 以加速最終訓練，視硬體資源而定
    final_training_args = TrainingArguments(
        output_dir="./final_model",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,  # 調低累積步數可加速訓練
        save_steps=1000,
        save_total_limit=1,
        logging_dir="./logs_final",
        logging_steps=50,
        evaluation_strategy="no",
        learning_rate=5e-4,
        fp16=True,
        report_to="none",
        dataloader_num_workers=0
    )
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.config.use_cache = False
    base_model.to(device)
    base_model.gradient_checkpointing_enable()
    model = get_peft_model(base_model, lora_config)
    trainer = Trainer(
        model=model,
        args=final_training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    save_path = r"C:\Users\ai\Desktop\春日部\fine tune\微調後的模型\Mistral\final_model"
    trainer.save_model(save_path)
    print(f"最終模型已保存至 {save_path}")

if __name__ == "__main__":
    main()
