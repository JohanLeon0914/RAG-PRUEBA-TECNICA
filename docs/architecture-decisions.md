# Architecture Decisions

## 1. BBVA -> Bancolombia

The original target was BBVA Colombia, but standard Python HTTP clients received HTTP 403. The assignment allowed using another bank, so Bancolombia was selected after validating that public pages could be accessed with normal HTTP requests. We did not implement CAPTCHA bypass, proxy rotation, browser fingerprint spoofing, or any antibot evasion.

## 2. Raw And Processed Data

The pipeline stores raw documents separately from processed documents. This makes it possible to improve cleaning, chunking, or evaluation without scraping the site again, which improves reproducibility and avoids unnecessary traffic.

## 3. Cleaner

The cleaner removes recurring navigation, footer, cookies, breadcrumbs, and UI noise while preserving product names, features, benefits, requirements, costs, conditions, and FAQ-like content. This keeps scraping and cleaning as separate responsibilities.

## 4. Curated Corpus

The evaluation corpus was built from sitemap/category discovery, then filtered to product-oriented pages for people. The current curated corpus has 40 CleanDocuments and 233 chunks. The home page is excluded from the RAG corpus because it is useful for navigation but has low density of product-specific knowledge.

## 5. Chunking

Chunking uses 900 characters with 150 characters overlap. This is a deterministic baseline, not a claim of optimality. The goal is to preserve enough local context while keeping chunks small enough for retrieval and future prompt construction.

## 6. Qdrant

Qdrant is used as the vector database because it is easy to run locally with Docker, supports cosine vector search, stores payload metadata, and maps cleanly to Qdrant Cloud later. The code accesses it through `VectorRepository`, so ingestion and RAG logic do not depend directly on `qdrant-client`.

## 7. BGE-M3 vs GTE

BGE-M3 and GTE Multilingual Base were compared on the same corpus, chunks, questions, and ground truth. BGE-M3 was selected for this corpus based on measured results: Hit@1 0.6000, Hit@5 0.8000, MRR 0.6806. This does not mean BGE-M3 is universally better.

## 8. Reranker

The BGE reranker improved retrieval quality: Hit@5 reached 0.9333 and MRR reached 0.7844. It also added about 5.18 seconds of CPU latency per query. For an interactive local chatbot, it is retained as an optional strategy but disabled by default with `RERANK_ENABLED=false`.

## 9. LLM Factory

Groq is the primary LLM provider and Gemini is kept as an alternative. `LLMFactory` creates the configured provider so `RAGService` does not import Groq, Gemini, or provider-specific SDKs.

## 10. Grounding And OOD UX

The LLM must return structured JSON with `answer` and `supported_by_context`. If `supported_by_context=false`, the API returns `sources=[]` even though Qdrant retrieved nearest neighbors. This is a UX and grounding decision, not a formal replacement for retrieval evaluation or a final relevance threshold.

## 11. Offline Ingestion vs Online Inference

Offline ingestion builds the knowledge base:

```text
Bancolombia -> Scraper -> Raw -> Cleaner -> CleanDocuments -> Chunker -> BGE -> Qdrant
```

Online inference answers user questions:

```text
User -> Frontend -> POST /chat -> RAGService -> BGE query embedding
     -> Qdrant -> optional reranker -> ContextBuilder -> LLM -> answer
```

Scraping and ingestion are not performed per user query.
