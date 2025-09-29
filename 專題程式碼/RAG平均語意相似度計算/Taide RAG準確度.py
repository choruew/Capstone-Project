import os
import json
import time
import torch
from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
from peft import PeftModel

# === 固定使用 GPU0，若無 GPU 則使用 CPU ===
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print("PyTorch 偵測到的裝置：", device)
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.cuda.device_count() =", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"裝置 {i}:", torch.cuda.get_device_name(i))

# === 測試 GPU 運算用函式 ===
def test_gpu_usage():
    if device.startswith("cuda"):
        print("\n[GPU測試] 正在進行大矩陣運算...")
        start_time = time.time()
        a = torch.randn((10000, 10000), device=device)
        b = torch.randn((10000, 10000), device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # 等待運算完成
        end_time = time.time()
        print("[GPU測試] 矩陣相乘完成，耗時:", f"{end_time - start_time:.2f} 秒")
        print("c.shape =", c.shape)
    else:
        print("\n[GPU測試] 當前為 CPU 模式，無法進行 GPU 測試。")

# === 設定檔案路徑 ===
pdf_root_folder = r"C:\Users\ai\Desktop\春日部\RAG準確度測試\已生成問題的資料"
faiss_db_path = r"C:\Users\ai\Desktop\faiss"
json_file_path = r"C:\Users\ai\Desktop\春日部\RAG準確度測試\驗證集.json"

# === 模型名稱 ===
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
similarity_model_name = "GanymedeNil/text2vec-large-chinese"

def clear_vector_database():
    """清理 FAISS 向量資料庫"""
    if os.path.exists(faiss_db_path):
        for file_name in os.listdir(faiss_db_path):
            file_path = os.path.join(faiss_db_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("✅ 向量資料庫已清理完畢。")

def extract_text_from_pdfs(pdf_root_folder):
    """從 PDF 中提取文本，並顯示進度條"""
    documents = []
    pdf_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(pdf_root_folder)
        for file in files if file.lower().endswith(".pdf")
    ]

    print(f"📂 找到 {len(pdf_files)} 個 PDF，開始讀取...")
    progress_bar = tqdm(total=len(pdf_files), desc="讀取 PDF", unit="file")

    for pdf_path in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"❌ 錯誤：無法讀取 {pdf_path}，原因：{e}")
        progress_bar.update(1)

    progress_bar.close()
    print(f"✅ 讀取完成，共提取 {len(documents)} 段文本。")
    return documents

def split_text(documents):
    """文本切割（加上進度條）"""
    print("✂️  正在切割文本...")  # 印出提示訊息表示開始切割文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=400)  # 使用遞迴式字元切割器，設定每段最大800字元，重疊400字元
    progress_bar = tqdm(total=len(documents), desc="文本切割", unit="段")  # 建立進度條，總數為文件數量，單位為「段」
    all_splits = []  # 儲存所有切割後的區塊
    for doc in documents:  # 逐一處理每個文件
        all_splits.extend(text_splitter.split_documents([doc]))  # 將切割結果加入 all_splits
        progress_bar.update(1)  # 更新進度條
    progress_bar.close()  # 關閉進度條
    print(f"✅ 文本切割完成，共產生 {len(all_splits)} 個區塊。")  # 印出完成訊息與區塊總數
    return all_splits  # 回傳切割後的所有區塊

def create_vector_database(documents):
    """創建 FAISS 向量資料庫（加上進度條）"""
    print("🔍 正在創建新的向量資料庫...")  # 印出提示訊息
    embedding = HuggingFaceEmbeddings(
        model_name=similarity_model_name,  # 指定要使用的語意嵌入模型
        encode_kwargs={"batch_size": 32},  # 設定批次大小為 32
        model_kwargs={'device': device}  # 設定模型執行的裝置（如 CPU 或 GPU）
    )

    progress_bar = tqdm(total=len(documents), desc="處理向量", unit="chunk")  # 建立進度條，用於追蹤向量處理進度
    all_chunks = []  # 儲存所有向量化處理過的區塊
    for i in range(0, len(documents), 32):  # 每 32 筆文件為一批進行處理
        batch_docs = documents[i:i+32]  # 取得當前批次的文件
        _ = embedding.embed_documents([doc.page_content for doc in batch_docs])  # 對每個文件內容進行嵌入（轉成向量）
        all_chunks.extend(batch_docs)  # 將本批次的文件加入 all_chunks
        progress_bar.update(len(batch_docs))  # 依照處理數量更新進度條
    progress_bar.close()  # 關閉進度條

    print("🔍 建立 FAISS 資料庫中...")  # 印出建立資料庫提示
    vectordb = FAISS.from_documents(all_chunks, embedding)  # 使用處理後的文件與嵌入模型建立 FAISS 向量資料庫
    vectordb.save_local(faiss_db_path)  # 將資料庫儲存到本機指定路徑
    print(f"✅ 向量資料庫儲存完成: {faiss_db_path}")  # 印出儲存成功訊息
    return vectordb  # 回傳建立好的 FAISS 向量資料庫物件


def setup_qa_system(vectordb):
    """設置 LLM 問答系統（確保在 GPU 上運行）"""
    print("🚀 載入基底模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": device},
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
    )
    model.to(device)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    # 注意：已透過 accelerate 載入模型，不要再指定 device
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=30,
        temperature=0.6,
        top_p=0.9
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    retriever = vectordb.as_retriever(search_kwargs={"k": 1})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def load_questions(json_file_path):
    """讀取 JSON 問題集"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def evaluate_model(qa_chain, json_file_path):
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
        batch_answers = [qa_chain.invoke({"query": p})["result"].strip() for p in batch_prompts]
        generated_answers.extend(batch_answers)

    reference_embeddings = sentence_model.encode(reference_answers, batch_size=16, convert_to_tensor=True)
    generated_embeddings = sentence_model.encode(generated_answers, batch_size=16, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(reference_embeddings, generated_embeddings).diagonal().tolist()
    avg_similarity = sum(similarities) / len(similarities)

    del reference_embeddings, generated_embeddings
    torch.cuda.empty_cache()

    print(f"✅ 平均相似度: {avg_similarity:.4f}")

def main():
    """主程序執行流程"""
    # 1. 測試 GPU 運算
    test_gpu_usage()

    # 2. 進行 RAG 流程
    start_time = time.time()
    clear_vector_database()
    
    documents = extract_text_from_pdfs(pdf_root_folder)
    if not documents:
        print("❌ 未找到任何 PDF 文件，程序結束。")
        return
    
    text_chunks = split_text(documents)
    vectordb = create_vector_database(text_chunks)
    qa_chain = setup_qa_system(vectordb)
    
    end_time = time.time()
    print(f"⏳ 總執行時間: {end_time - start_time:.2f} 秒")
    
    evaluate_model(qa_chain, json_file_path)

if __name__ == "__main__":
    main()
