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
from peft import PeftModel  # 新增載入 PEFT 模組


# 設定檔案路徑
pdf_root_folder = r"C:\Users\ai\Desktop\春日部\RAG準確度測試\已生成問題的資料"
faiss_db_path = r"C:\Users\ai\Desktop\Taiwan Llama faiss"
json_file_path = r"C:\Users\ai\Desktop\春日部\RAG準確度測試\驗證集.json"

# 設定模型
base_model_name = "yentinglin/Llama-3-Taiwan-8B-Instruct"
# 請將此路徑換成您微調後的權重資料夾路徑
finetuned_weights_path = r"C:\Users\ai\Desktop\春日部\fine tune\微調後的模型\Taiwan-Llama-3\final_model"
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
    pdf_files = [os.path.join(root, file)
                 for root, _, files in os.walk(pdf_root_folder)
                 for file in files if file.lower().endswith(".pdf")]

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
    print("✂️  正在切割文本...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=400)

    progress_bar = tqdm(total=len(documents), desc="文本切割", unit="段")
    all_splits = []
    
    for doc in documents:
        all_splits.extend(text_splitter.split_documents([doc]))
        progress_bar.update(1)
    
    progress_bar.close()
    print(f"✅ 文本切割完成，共產生 {len(all_splits)} 個區塊。")
    return all_splits

def create_vector_database(documents):
    """創建 FAISS 向量資料庫（加上進度條）"""
    print("🔍 正在創建新的向量資料庫...")

    embedding = HuggingFaceEmbeddings(
        model_name=similarity_model_name,
        encode_kwargs={"batch_size": 32},  # 批量處理
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    # 分批處理嵌入，並更新進度條
    progress_bar = tqdm(total=len(documents), desc="處理向量", unit="chunk")
    all_chunks = []
    for i in range(0, len(documents), 32):  # 32 為 batch size
        batch_docs = documents[i:i+32]
        _ = embedding.embed_documents([doc.page_content for doc in batch_docs])
        all_chunks.extend(batch_docs)
        progress_bar.update(len(batch_docs))
    progress_bar.close()

    # 建立 FAISS 向量資料庫
    print("🔍 建立 FAISS 資料庫中...")
    vectordb = FAISS.from_documents(all_chunks, embedding)
    vectordb.save_local(faiss_db_path)
    print(f"✅ 向量資料庫儲存完成: {faiss_db_path}")
    return vectordb

def setup_qa_system(vectordb):
    """設置 LLM 問答系統，使用微調後並整合 RAG 的模型"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        device_map="auto",  
        offload_folder="offload",  
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    # 載入微調後的權重
    model = PeftModel.from_pretrained(model, finetuned_weights_path)

    tokenizer.pad_token_id = tokenizer.eos_token_id  
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=30,  
        do_sample=True,
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
    """使用 JSON 問題集測試問答系統準確度，生成後立即印出問題與生成答案"""
    questions = load_questions(json_file_path)
    sentence_model = SentenceTransformer(similarity_model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    questions_list = [entry["question"] for entry in questions]
    reference_answers = [entry["answer"] for entry in questions]

    generated_answers = []
    for q in questions_list:
        prompt = f"請以繁體中文回答以下問題：\n{q}"
        result = qa_chain.invoke({"query": prompt})["result"].strip()
        print("問題：", q)
        print("生成答案：", result)
        print("-" * 50)
        generated_answers.append(result)

    # 計算語意相似度
    reference_embeddings = sentence_model.encode(reference_answers, batch_size=16, convert_to_tensor=True)
    generated_embeddings = sentence_model.encode(generated_answers, batch_size=16, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(reference_embeddings, generated_embeddings).diagonal().tolist()
    avg_similarity = sum(similarities) / len(similarities)
    print(f"✅ 平均相似度: {avg_similarity:.4f}")

    del reference_embeddings, generated_embeddings
    torch.cuda.empty_cache()



def main():
    """主程序執行流程"""
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
