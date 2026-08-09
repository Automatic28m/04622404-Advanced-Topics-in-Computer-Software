import os
import sys

import config
from src import index_meta
from src.rag_pipeline import RAGPipeline

def print_answer(result):
    print()
    print(f"💡 Answer: {result['answer']}")

    if config.SHOW_DEBUG:
        print(f"\n[Debug] Search queries: {result['queries_used']}")
        print(f"[Debug] Execution time (seconds): {result['timings']}")

exit_code = ["exit", "quit", "q"]
def print_exit():
    print("Enter ", end='')
    count = 0
    for i in exit_code:
        print(f"\"{i}\" ", end='')
        if count+1 != len(exit_code):
            print(", ", end='')
            count+=1
    print("To exit")

def main():
    print("--- RAG System for Phanlop's Portfolio ---")
    print_exit()
    
    if not os.path.exists(config.FAISS_INDEX_FILE):
        print("Vector database not found.")
        print("Run: python build_index.py")
        return

    # Check if index needs rebuild
    index_meta.warn_if_stale()

    rag = RAGPipeline()
    # rag.show_settings()

    print("\n👋Hello! This is Phanlop Boonluea")
    print("\nThis is my portfolio website https://phanlopboonluea.netlify.app/")
    print("\nIf you have any doubt in me, feel free to ask!")
    # print("==="*50)
    # print("\n💡ตัวอย่างคำถาม")
    # print("\n- มีทักษะอะไรบ้าง")
    # print("\n- ทักษะด้านภาษาอังกฤษ")
    # print("\n- ทักษะด้านเอไอ")
    # print("\n- ทักษะด้าน IoT")
    # print("\n- รางวัลและความสำเร็จ")
    # print("\n- ทุนที่เคยได้รับ")
    # print("==="*50)

    while True:
        query = input("\n💬 Question: ").strip()

        if query.lower() in exit_code:
            print("See you later!")
            break

        if not query:
            continue

        result = rag.ask(query)
        print_answer(result)
        print("==="*50)

if __name__ == "__main__":
    main()
