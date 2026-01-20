import os
from src.build_index import rebuild_index
from src.query_engine import VisaRAG

def ensure_index():
    if not os.path.exists("index/faiss_index.bin"):
        print("Building index")
        rebuild_index("data")
        print("Index build completed")
    else:
        print("Index already exists")

def run_confirmation_pipeline():
    rag = VisaRAG()
    rag.run_test()
    print("Pipeline executed successfully")

if __name__ == "__main__":
    ensure_index()
    run_confirmation_pipeline()
