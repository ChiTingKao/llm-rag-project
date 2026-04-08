import os
import configparser

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 專案根目錄

config = configparser.ConfigParser()
config.read(os.path.join(BASE_DIR, "config.ini"), encoding="utf-8")

# ---------------- General ----------------
CHUNKS_PATH = config["general"]["chunks_path"]
FAISS_INDEX_PATH = config["general"]["faiss_index_path"]

# ---------------- Data ----------------
JSON_PATH = config["data"]["json_path"]

# ---------------- Retrieval ----------------
HYBRID_ALPHA = float(config["retrieval"]["hybrid_alpha"])
TOP_K = int(config["retrieval"]["top_k"])

# ---------------- LLM ----------------
LLM_MODEL = config["llm"]["model"]
LLM_ENDPOINT = config["llm"]["endpoint"]
STREAM = config["llm"].getboolean("stream")

# ---------------- Test CSV ----------------
TEST_CSV = config["test"]["test_csv"]
EVAL_CSV = config["test"]["eval_csv"]

