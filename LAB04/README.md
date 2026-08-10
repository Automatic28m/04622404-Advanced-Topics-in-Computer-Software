
## DL-04-RAG System Development I

Build a complete RAG system in Python, from loading a knowledge base and splitting text into chunks to retrieving relevant information and generating answers.

---- 
# Structure

```text
RAG-Project/
│
├── data/
│   ├── sex_q_a.txt
│   └── golden_set.json                     # Evaluation set
│
├── outputs/
│   ├── extracted_text.json                 # Parsed Q&A pairs with line numbers
│   ├── chunks.json                         # 541 text chunks with metadata
│   ├── embeddings.npy                      # Embedding vectors
│   ├── retrieval_results.json              # Top-k retrieval results
│   ├── eval_retrieval.json                 ⭐ Retrieval scores per configuration
│   └── eval_generation.json                ⭐ Answer quality scores
│
├── vector_db/
│   ├── document.index                      # FAISS index — dense semantic search
│   ├── bm25_index.pkl                      ⭐ BM25 index — exact-token search
│   ├── chunk_store.json                    # Chunks + metadata, aligned with FAISS order
│   └── index_meta.json                     ⭐ Fingerprint of the dataset this index was built from
│
├── labs/
│   ├── lab01_extract_text.py               # Extract text from the source file
│   ├── lab02_chunking.py                   # Split text into chunks
│   ├── lab03_create_embeddings.py          # Generate embeddings
│   ├── lab04_create_vector_db.py           # Build the FAISS vector database
│   ├── lab05_query_embedding.py            # Create query embeddings
│   ├── lab06_similarity_search.py          # Retrieve top-k relevant chunks
│   └── lab07_complete_retrieval.py         # Complete retrieval pipeline
│
├── src/
│   ├── document_loader.py                  # File loading and text extraction
│   ├── text_splitter.py                    # Text chunking
│   ├── embedding_model.py                  # Embedding model
│   ├── vector_store.py                     # FAISS vector database
│   ├── index_meta.py                       ⭐ Detect when the index is stale vs the dataset
│   ├── retriever.py                        # Dense-only retrieval
│   ├── hybrid_retriever.py                 ⭐ BM25 + Dense + RRF Fusion
│   ├── rerankers.py                        ⭐ Cross-Encoder Reranking
│   ├── query_transform.py                  ⭐ Query Rewrite, Multi-Query, HyDE
│   ├── prompt_templates.py                 ⭐ Prompt Templates
│   ├── generator.py                        ⭐ LLM Answer Generation
│   ├── memory.py                           ⭐ Conversation History
│   └── rag_pipeline.py                     ⭐ End-to-End RAG Pipeline
│
├── evaluation/
│   ├── metrics.py                          ⭐ Hit@k, Recall@k, Precision@k, MRR, nDCG
│   ├── build_golden_set.py                 ⭐ Generate the evaluation set
│   ├── eval_retrieval.py                   ⭐ Compare retrieval configurations
│   └── eval_generation.py                  ⭐ Evaluate answer quality
│
├── config.py                               # Project configuration
├── build_index.py                          # Build all indexes
└── main.py                                 # Run the RAG system
```

## Summary

This project implements an Advanced Retrieval-Augmented Generation (RAG) system specifically designed as an AI assistant for a professional portfolio. It extracts data from personal Github repositories and professional documents, splits the text into chunks, generates embeddings, and stores them in a local FAISS vector database.

The system utilizes advanced RAG techniques to improve retrieval quality, including **BM25 keyword search, hybrid retrieval with Reciprocal Rank Fusion (RRF), cross-encoder reranking, and query transformation**. It features a robust generation pipeline that uses LLMs to synthesize answers with accurate citations and maintains conversation memory for follow-up questions.

All features can be enabled or disabled in **config.py**, making it easy to experiment with different architectures. The project also includes an automated evaluation module that tests retrieval performance using **Hit@k, MRR, and nDCG**, and uses an LLM-as-a-Judge to measure the faithfulness and correctness of generated answers.

The knowledge base contains personal portfolio data, professional experience, and GitHub repository metadata. It supports **Groq, OpenAI, and Gemini** through a configurable interface.

## How to Evaluate the RAG System

This project includes an automated evaluation suite to measure the accuracy of both your search engine (retrieval) and the LLM answers (generation).

### Step 1: Build the Golden Set (Test Data)
First, generate the test dataset. This script uses your data chunks to create 4 variations of test questions (verbatim, slang, partial, natural) to simulate real user queries.
```bash
python -m evaluation.build_golden_set
```
* **What to look for:** It will output an example of the generated question variants and save the "answer key" to `data/portfolio_golden_set.json`.

### Step 2: Evaluate Retrieval Quality (Search Accuracy)
Test how accurately your system finds the correct chunks without relying on the LLM. 
```bash
python -m evaluation.eval_retrieval
```
* **What to look for:** 
  * **Hit@10:** Did the correct chunk appear anywhere in the top 10 results? (e.g., `1.0000` means 100% success on that query type).
  * **MRR (Mean Reciprocal Rank):** How high up in the results was the correct chunk? Closer to 1.0 is better.
  * **Best configuration:** The summary at the bottom tells you which search method performed best. For example, `hybrid` (BM25 + Dense) usually beats `dense_only` (pure vector search).

### Step 3: Evaluate Generation Quality (LLM Answers)
Test if the LLM is correctly summarizing the retrieved chunks, or if it is hallucinating.
```bash
python -m evaluation.eval_generation
```
* **What to look for:**
  * **Faithfulness / Correctness:** Scores closer to 1.0 mean the LLM strictly followed your source documents and answered correctly.
  * **Least Faithful Answers:** The script highlights answers with low faithfulness scores (e.g., `< 0.60`), allowing you to easily spot where the LLM might be "hallucinating" or making up information not found in your portfolio.

### Evaluation Logic (Pseudocode)
Here is a simplified breakdown of how the evaluation pipeline works under the hood:

```python
# 1. build_golden_set.py
function BuildGoldenSet(database_chunks):
    test_cases = []
    for chunk in database_chunks:
        # Use LLM to generate test questions that SHOULD match this chunk
        questions = LLM.generate(
            "Create 4 question variants: verbatim, slang, partial, natural"
        )
        test_cases.append({
            "question_variants": questions,
            "expected_chunk_id": chunk.id
        })
    save_to_json(test_cases)

# 2. eval_retrieval.py
function EvaluateRetrieval(test_cases, search_methods):
    for method in search_methods:  # e.g., Dense, BM25, Hybrid, Hybrid+Rerank
        for test in test_cases:
            results = method.search(test.question)
            
            # Check if our expected chunk was found by the search engine
            if test.expected_chunk_id in results.top(10):
                score.hit_10 += 1
                score.mrr += calculate_rank_position(results, test.expected_chunk_id)
                
    print("Best Retrieval Method:", get_highest(score.mrr))

# 3. eval_generation.py
function EvaluateGeneration(test_cases, rag_pipeline):
    for test in test_cases:
        # Generate an actual answer using the full RAG system
        llm_answer = rag_pipeline.ask(test.question)
        
        # Use a "Judge LLM" prompt to grade the generated answer
        faithfulness = JudgeLLM.grade("Does this answer strictly use the source text?")
        correctness = JudgeLLM.grade("Does this answer match the expected meaning?")
        
        score.faithfulness += faithfulness
        score.correctness += correctness
```
