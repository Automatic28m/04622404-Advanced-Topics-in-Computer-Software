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
      #  print("Vector database not found.")
      #  print("Please run lab01_extract_text.py -> lab04_create_vector_db.py first.")
        return

    retriever = Retriever(
        model_name=config.EMBEDDING_MODEL_NAME,
        index_path=config.FAISS_INDEX_FILE,
        chunk_store_path=config.CHUNK_STORE_FILE,
    )

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
        print_exit()
        query = input("\n💬 คำถาม : ").strip()

        if query.lower() in ("ออก", "exit", "quit", "q"):
            print("แล้วเจอกันใหม่!")
            break

        if not query:
            continue

        results = retriever.retrieve(query)

        if not results:
            print("ไม่มีคำตอบที่เกี่ยวข้องในคลังความรู้")
            print("==="*50)
            continue

        for rank, item in enumerate(results, start=1):
            print_answer(rank, item)
        print("==="*50)


if __name__ == "__main__":
    main()
