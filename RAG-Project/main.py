from src.retriever import Retriever
import config
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def print_answer(rank, item):
    print(f"\n💡ผลลัพธ์ที่ {rank} (คะแนน: {item['score']:.2f})")
    # print(f"Category: {item['category']}")
    # print(f"Similar Question: {item['question']}")
    print(f"คำตอบ: {item['answer']}")


def main():
    if not os.path.exists(config.FAISS_INDEX_FILE):
      #  print("Vector database not found.")
      #  print("Please run lab01_extract_text.py -> lab04_create_vector_db.py first.")
        return

    print("-ระบบ RAG เพื่อตอบคำถามเกี่ยวกับแฟ้มสะสมผลงาน---")
    print("-Enter ('exit', 'quit', หรือ 'q' เพื่อออก)---\n")

    retriever = Retriever(
        model_name=config.EMBEDDING_MODEL_NAME,
        index_path=config.FAISS_INDEX_FILE,
        chunk_store_path=config.CHUNK_STORE_FILE,
    )

    print("\n👋สวัสดีครับ! ผมพัลลภ บุญเหลือ")
    print("\n👋เว็บไซต์แฟ้มสะสมผลงาน https://phanlopboonluea.netlify.app/")
    print("\n👋สงสัยเพิ่มเติมอะไร ถามได้เลยครับ")
    while True:
        query = input("\nคำถาม : ").strip()

        if query.lower() in ("exit", "quit", "q"):
            print("See you later!")
            break

        if not query:
            continue

        results = retriever.retrieve(query)

        if not results:
            print("ไม่มีคำตอบที่เกี่ยวข้องในคลังความรู้")
            continue

        for rank, item in enumerate(results, start=1):
            print_answer(rank, item)


if __name__ == "__main__":
    main()
