import time
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
from langchain_huggingface import HuggingFacePipeline

# === 固定使用 GPU0，若無 GPU 則使用 CPU ===
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("PyTorch 偵測到的裝置：", device)

# === 模型名稱 ===
model_name = "yentinglin/Llama-3-Taiwan-8B-Instruct"
similarity_model_name = "GanymedeNil/text2vec-large-chinese"

# === 問題集路徑 ===
json_file_path = r"C:\Users\ai\Desktop\春日部\RAG準確度測試\驗證集.json"

def setup_llm_pipeline():
    """載入 LLM 模型與生成管線"""
    print("🚀 載入基底模型中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": device},
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
    )
    model.to(device)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=30,
        temperature=0.6,
        top_p=0.9,
        do_sample=True  # 啟用抽樣才會使用 temperature / top_p
    )
    return HuggingFacePipeline(pipeline=text_generation_pipeline)

def load_questions(json_file_path):
    """讀取 JSON 問題集"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def evaluate_model(llm, json_file_path):
    """使用 JSON 問題集測試問答系統準確度"""
    questions = load_questions(json_file_path)
    sentence_model = SentenceTransformer(similarity_model_name, device=device)

    questions_list = [entry["question"] for entry in questions]
    reference_answers = [entry["answer"] for entry in questions]
    prompts = [f"請以繁體中文回答以下問題：\n{q}" for q in questions_list]

    batch_size = 5
    generated_answers = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_outputs = [llm.invoke(p).strip() for p in batch_prompts]  # 修正點：回傳值為字串
        generated_answers.extend(batch_outputs)

    reference_embeddings = sentence_model.encode(reference_answers, batch_size=16, convert_to_tensor=True)
    generated_embeddings = sentence_model.encode(generated_answers, batch_size=16, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(reference_embeddings, generated_embeddings).diagonal().tolist()
    avg_similarity = sum(similarities) / len(similarities)

    print(f"✅ 平均語意相似度: {avg_similarity:.4f}")
    return avg_similarity

def main():
    start_time = time.time()
    llm = setup_llm_pipeline()
    evaluate_model(llm, json_file_path)
    end_time = time.time()
    print(f"⏳ 執行總時間: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
