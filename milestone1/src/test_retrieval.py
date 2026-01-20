# src/test_retrieval.py

from typing import List, Dict

from rag_pipeline import retrieve_relevant_chunks


def pretty_print_chunks(chunks: List[Dict]) -> None:
    for i, ch in enumerate(chunks, start=1):
        preview = ch["text"][:300].replace("\n", " ")
        print(f"\n----- RESULT {i} -----")
        print(f"Source: {ch.get('source', 'unknown')}")
        print(f"Score (L2 distance): {ch.get('score'):.4f}")
        print(f"Text preview: {preview}...")
        print("-----------------------")


def main():
    print("🔍 FAISS Retrieval Test")
    question = input("Enter a test question: ")

    chunks = retrieve_relevant_chunks(question, top_k=5)
    if not chunks:
        print("No chunks retrieved. Check if FAISS index and chunks are built.")
        return

    pretty_print_chunks(chunks)


if __name__ == "__main__":
    main()
