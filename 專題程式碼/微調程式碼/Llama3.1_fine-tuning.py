import os
import json
import torch
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

# 合併問題與答案
def preprocess_function(examples, tokenizer, max_length=48):
    inputs = ["Q: " + q + "\nA:" for q in examples['question']]
    targets = [a for a in examples['answer']]

    # 確保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length")["input_ids"]

    # 避免標籤填充影響 loss 計算
    model_inputs["labels"] = [
        label + [-100] * (len(model_inputs["input_ids"][i]) - len(label))
        if len(label) < len(model_inputs["input_ids"][i])
        else label[:len(model_inputs["input_ids"][i])]
        for i, label in enumerate(labels)
    ]
    return model_inputs

# 主程式
def main():
    # 配置 CUDA 動態分配
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
    torch.backends.cuda.matmul.allow_tf32 = True  # 啟用 TensorFloat32

    # 檢查 CUDA 是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 載入資料
    file_path = r"C:\Users\ai\Desktop\春日部\fine tune\金管會問題集 JSON格式.json"
    raw_data = load_data(file_path)
    dataset = prepare_dataset(raw_data)

    # 初始化分詞器
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 設定 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # LoRA 配置
    lora_config = LoraConfig(
        r=8,  # 降低 LoRA 矩陣的秩（rank）
        lora_alpha=16,  # 調整 scaling factor
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # K 折交叉驗證
    k = 3
    kfold = KFold(n_splits=k)
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
            num_train_epochs=5,  # 減少訓練次數
            per_device_train_batch_size=1,
            gradient_accumulation_steps=96,  # 增加梯度累積步數
            save_steps=1000,
            save_total_limit=1,
            logging_dir=f"./logs_fold_{fold + 1}",
            logging_steps=50,
            evaluation_strategy="epoch",
            learning_rate=5e-4,
            fp16=True,
            report_to="none"
        )

        # 初始化模型並應用 LoRA
        base_model = LlamaForCausalLM.from_pretrained(model_name).to(device)
        base_model.gradient_checkpointing_enable()  # 啟用梯度檢查點
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

        # 訓練器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        # 開始訓練
        trainer.train()
        eval_result = trainer.evaluate()
        print(f"第 {fold + 1} 折評估結果: {eval_result}")
        fold_results.append(eval_result)

    # 完整資料集微調
    print("\n=== 完成 K 折，進行最終微調 ===")
    training_args = TrainingArguments(
        output_dir="./final_model",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=96,
        save_steps=1000,
        save_total_limit=1,
        logging_dir="./logs_final",
        logging_steps=50,
        evaluation_strategy="no",
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
    save_path = r"C:\Users\ai\Desktop\春日部\fine tune\微調後的模型\Llama3.1-2\final_model"
    trainer.save_model(save_path)
    print(f"最終模型已保存至 {save_path}")

if __name__ == "__main__":
    main()