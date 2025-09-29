本研究旨在提升金融法規問答系統的效能，透過比較四種不同的大型語言模型（TAIDE、Taiwan-Llama、Mistral、Llama3.1），探討檢索增強生成（Retrieval-Augmented Generation, RAG）技術與微調（Fine-tuning）方法對模型準確性的影響。
研究過程包括：
（1）對金融法規相關PDF文件進行結構化處理，建立向量資料庫以支援RAG技術
（2）透過基礎RAG測試評估四種模型的初始準確性
（3）基於高品質問題集進行LoRA（Low-Rank Adaptation）微調
（4）評估經微調後的四種大型語言模型在結合RAG技術時的最終表現，並比較其在問答準確度、檢索效率與應用價值上的提升幅度。
