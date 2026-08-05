import os
import sys

import config
from src import index_meta
from src.rag_pipeline import RAGPipeline

def print_answer(result):
    print()
    print(f"💡 คำตอบ: {result['answer']}")

    if config.SHOW_DEBUG:
        print(f"\n[Debug] Search queries: {result['queries_used']}")
        print(f"[Debug] Execution time (seconds): {result['timings']}")

def print_exit():
    exit_code = ["ออก", "exit", "quit", "q"]
    print("💡พิมพ์ ", end='')
    count = 0
    for i in exit_code:
        print(f"\"{i}\" ", end='')
        if count+1 != len(exit_code):
            print(", ", end='')
            count+=1
    print("เพื่อออก")

def main():
    print("-ระบบ RAG เพื่อตอบคำถามเกี่ยวกับแฟ้มสะสมผลงาน---")
    print_exit()
    
    if not os.path.exists(config.FAISS_INDEX_FILE):
        print("Vector database not found.")
        print("Run: python build_index.py")
        return

    # Check if index needs rebuild
    index_meta.warn_if_stale()

    rag = RAGPipeline()
    # rag.show_settings()

    print("\n👋สวัสดีครับ! ผมพัลลภ บุญเหลือ")
    print("\nเว็บไซต์แฟ้มสะสมผลงาน https://phanlopboonluea.netlify.app/")
    print("\nสงสัยเพิ่มเติมอะไร ถามได้เลยครับ")
    print("==="*50)
    print("\n💡ตัวอย่างคำถาม")
    print("\n- มีทักษะอะไรบ้าง")
    print("\n- ทักษะด้านภาษาอังกฤษ")
    print("\n- ทักษะด้านเอไอ")
    print("\n- ทักษะด้าน IoT")
    print("\n- รางวัลและความสำเร็จ")
    print("\n- ทุนที่เคยได้รับ")
    print("==="*50)

    while True:
        query = input("\n💬 คำถาม : ").strip()

        if query.lower() in ("ออก", "exit", "quit", "q"):
            print("แล้วเจอกันใหม่!")
            break

        if not query:
            continue

        result = rag.ask(query)
        print_answer(result)
        print("==="*50)

if __name__ == "__main__":
    main()
