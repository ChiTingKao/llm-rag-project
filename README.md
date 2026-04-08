### 簡介
#### LLM Python QA RAG System
#### 這是一個基於 FAISS + SentenceTransformer + CrossEncoder + Ollama 的 RAG（Retrieval-Augmented Generation）問答系統，專門針對 Python 教學資料進行問答。
#### 使用者可以輸入 Python 問題，系統會自動檢索相關內容、重排序，並生成答案。
#### 
#### 
### 專案結構
#### LLM/
#### ├─ main.py                # 主程式入口
#### ├─ config.ini             # 設定檔
#### ├─ evaluate               # 測試並對LLM回答進行評分
#### ├─ README.md
#### ├─ data/
#### │  ├─ processed/          # 向量索引及 chunk 資料
#### │  └─ test_data.csv       # 測試資料
#### └─ src/
####    ├─ __init__.py
####    ├─ config.py           # 配置讀取
####    ├─ build_index.py      # 建立向量庫
####    ├─ pipeline.py         # RAG 系統主程式
####    └─ evaluation          # 評分
#### 
### 使用說明
#### 1. 建立向量庫
#### python src/build_index.py
#### 會讀取 data/processed/python_tutorial.json
#### 將資料切成 chunk、生成向量，並存成 FAISS index 和 chunks
#### 
#### 2. 啟動問答
#### 在 main.py 裡修改問題：
#### question = "Python 如何使用 for 迴圈？"
#### 
#### 3. 測試資料問答
#### python evaluate.py 可以對 data/test_data.csv 批量測試
#### 
####     
### 配置參數
#### 在 config.ini 或 config.py 設定：
#### 
#### 
### 注意事項
#### FAISS 在 Windows 上對中文或長路徑敏感，建議將專案放在全英文路徑
#### 請先啟動 Ollama 或其他 LLM API
