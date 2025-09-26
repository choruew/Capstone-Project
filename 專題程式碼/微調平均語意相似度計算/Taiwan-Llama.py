import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

# 模型與檔案路徑設定
base_model_name = "yentinglin/Llama-3-Taiwan-8B-Instruct"
lora_weights_path = r"C:\Users\ai\Desktop\春日部\fine tune\微調後的模型\Taiwan-Llama-3\final_model"
json_file_path = r"C:\Users\ai\Desktop\春日部\RAG準確度測試\驗證集.json"
similarity_model_name = "GanymedeNil/text2vec-large-chinese"

# 載入 tokenizer 與基礎模型
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_8bit=True,  # 未來建議改用 quantization_config
    device_map="auto"
)

# 載入 LoRA 微調權重
model = PeftModel.from_pretrained(model, lora_weights_path)

# 載入語意相似度模型
similarity_model = SentenceTransformer(similarity_model_name)

# 載入驗證集 JSON 檔案，該檔案中每筆資料包含 "question" 與 "answer" 欄位
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

similarity_scores = []

# 逐筆處理驗證集資料
for idx, item in enumerate(data):
    # 檢查 JSON 資料結構，確保有 'question' 與 'answer' 欄位
    if "question" not in item or "answer" not in item:
        print(f"第 {idx} 筆資料缺少 'question' 或 'answer' 欄位，實際欄位有: {list(item.keys())}")
        continue

    prompt = item["question"]
    expected = item["answer"]

    # 使用 generate 產生答案
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(input_ids, max_length=200, do_sample=False)
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 取得生成答案與預期答案的句向量
    emb_generated = similarity_model.encode(generated, convert_to_tensor=True)
    emb_expected = similarity_model.encode(expected, convert_to_tensor=True)

    # 計算餘弦相似度
    cosine_sim = util.cos_sim(emb_generated, emb_expected).item()
    similarity_scores.append(cosine_sim)

    print("題目：", prompt)
    print("正確答案：", expected)
    print("生成答案：", generated)
    print("語意相似度：{:.4f}".format(cosine_sim))
    print("-" * 50)

if similarity_scores:
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    print("平均語意相似度：{:.4f}".format(avg_similarity))
else:
    print("沒有符合條件的資料進行評估。")
