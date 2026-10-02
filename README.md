# Bancolombia Conversational RAG

Asistente conversacional RAG sobre informacion publica de `bancolombia.com`.

El proyecto construye un flujo local completo: scraping controlado, limpieza, corpus curado, chunking, embeddings locales, Qdrant, retrieval denso, reranking opcional, generacion grounded con LLM, memoria conversacional persistida en SQLite, analytics runtime y frontend React/Vite. La prioridad del diseño es que cada decision sea reproducible localmente y defendible en una entrevista tecnica.

## Features

- Web scraping controlado de paginas publicas de Bancolombia.
- Almacenamiento separado de documentos raw y processed.
- Cleaner para reducir navegacion, footer, cookies, breadcrumbs y ruido repetitivo.
- Corpus curado de productos bancarios para personas, construido desde sitemap y categorias.
- Chunking deterministico con `CHUNK_SIZE=900` y `CHUNK_OVERLAP=150`.
- Embeddings locales open-source con BGE-M3 o GTE Multilingual.
- Qdrant como vector database local con volumen persistente.
- Retrieval denso con cosine similarity.
- Reranking opcional con CrossEncoder BGE.
- RAG grounded con fuentes reales provenientes de metadata.
- Salida estructurada del LLM con `supported_by_context`.
- Memoria conversacional por `session_id`.
- Persistencia de conversaciones en SQLite.
- Analytics runtime sobre el historial persistido.
- Frontend React/Vite con acceso a conversaciones anteriores.
- Ejecucion local reproducible con Docker Compose.

## Architecture

El sistema separa explicitamente ingestion offline de inferencia online. El scraping y la indexacion no ocurren durante cada pregunta del usuario.

Offline ingestion:

```text
Bancolombia
    ↓
Scraper
    ↓
Raw Documents
    ↓
Cleaner
    ↓
Processed Documents
    ↓
Corpus Selection
    ↓
Chunker
    ↓
Embedding Strategy
    ↓
Qdrant
```

Online inference:

```text
User
    ↓
React/Vite
    ↓
FastAPI
    ↓
RAGService
    ↓
ConversationRepository / SQLite history
    ↓
RetrievalPipeline
    ↓
BGE-M3
    ↓
Qdrant
    ↓
Optional Reranker
    ↓
ContextBuilder
    ↓
PromptBuilder + Conversation History
    ↓
LLM
    ↓
Grounded Answer + Sources
    ↓
SQLite persistence
```

Pipeline sin reranking, configuracion por defecto:

```text
Question -> BGE-M3 -> Qdrant cosine search -> Top 5 chunks
         -> ContextBuilder -> PromptBuilder + history -> LLM
         -> grounded answer + sources
```

Pipeline con reranking opcional:

```text
Question -> BGE-M3 -> Qdrant Top 15 candidates
         -> BGE CrossEncoder reranker -> Top 5 chunks
         -> ContextBuilder -> PromptBuilder + history -> LLM
         -> grounded answer + sources
```

Retrieval no es generation: retrieval recupera fragmentos similares; generation redacta la respuesta usando solo el contexto recuperado y el historial conversacional relevante. Qdrant puede devolver vecinos incluso para preguntas fuera del dominio, por eso el LLM debe declarar si la respuesta esta soportada por el contexto.

## Quick Start

Prerequisites:

- Docker
- Docker Compose v2

Clonar el repositorio y entrar al proyecto:

```bash
git clone https://github.com/JohanLeon0914/RAG-PRUEBA-TECNICA.git
cd RAG-PRUEBA-TECNICA
```

Configurar variables:

```bash
cp .env.example .env
```

Editar `.env` y agregar al menos:

```bash
GROQ_API_KEY=...
```

Levantar Qdrant:

```bash
docker compose up -d qdrant
```

Ejecutar ingestion offline:

```bash
docker compose run --rm api python -m scripts.ingest
```

Esto usa `data/corpus/processed_documents.jsonl`, genera chunks, calcula embeddings BGE-M3 y hace upsert en la coleccion configurada. No ejecuta scraping automatico.

Levantar el stack completo:

```bash
docker compose up --build
```

URLs locales:

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Qdrant: `http://localhost:6333`

La primera ejecucion puede descargar BGE-M3 si el volumen `hf_cache` esta vacio. La descarga queda cacheada para ejecuciones posteriores.

## Configuration

Variables principales:

```bash
CHUNK_SIZE=900
CHUNK_OVERLAP=150

EMBEDDING_PROVIDER=bge
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_BATCH_SIZE=16
EMBEDDING_DEVICE=cpu

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=bancolombia_eval_bge_m3

RETRIEVAL_TOP_K=10
RETRIEVAL_CANDIDATE_K=15
RERANK_ENABLED=false
RERANK_TOP_K=5
RAG_CONTEXT_TOP_K=5
RAG_MIN_RETRIEVAL_SCORE=

CONVERSATION_HISTORY_N_MESSAGES=6
CONVERSATION_DB_PATH=data/conversations/conversations.db

LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=
GEMINI_API_KEY=
```

Dentro de Docker Compose, la API usa `QDRANT_URL=http://qdrant:6333` y `CONVERSATION_DB_PATH=/app/data/conversations/conversations.db`. Desde host, `QDRANT_URL=http://localhost:6333`.

`CHUNK_SIZE` y `CHUNK_OVERLAP` estan medidos en caracteres, no tokens. Son un baseline configurable, no una afirmacion de optimalidad universal.

## API Usage

Crear o continuar conversacion:

```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"¿Qué opciones ofrece Bancolombia para financiar vivienda?","session_id":null}'
```

Respuesta:

```json
{
  "session_id": "uuid",
  "answer": "...",
  "supported_by_context": true,
  "sources": [
    {
      "title": "...",
      "url": "https://www.bancolombia.com/...",
      "chunk_index": 0,
      "score": 0.82
    }
  ],
  "metadata": {
    "embedding_latency_ms": 123.4,
    "search_latency_ms": 45.6,
    "llm_latency_ms": 789.0,
    "total_latency_ms": 1000.0
  }
}
```

Continuar usando la misma sesion:

```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"¿Y cuál sirve para remodelar?","session_id":"uuid"}'
```

Listar sesiones persistidas:

```bash
curl -s http://127.0.0.1:8000/sessions
```

Consultar mensajes de una sesion:

```bash
curl -s http://127.0.0.1:8000/sessions/uuid/messages
```

Consultar analytics runtime:

```bash
curl -s http://127.0.0.1:8000/analytics/summary
```

## Design Patterns

### Repository Pattern

```text
VectorRepository
    ↑
QdrantVectorRepository

ConversationRepository
    ↑
SQLiteConversationRepository
```

Problema real: la logica RAG necesita almacenamiento vectorial y persistencia conversacional, pero no debe conocer detalles de `qdrant-client` ni de SQL.

`VectorRepository` evita que `Retriever`, `RetrievalPipeline`, `RAGService` o ingestion dependan directamente de Qdrant. Esto permite cambiar entre Qdrant local y Qdrant Cloud mediante configuracion.

`ConversationRepository` evita que `RAGService` ejecute SQL directamente. SQLite puede reemplazarse posteriormente por PostgreSQL u otro store sin reescribir la orquestacion RAG.

### Strategy Pattern

```text
EmbeddingStrategy
├── BGEM3EmbeddingStrategy
└── GTEMultilingualEmbeddingStrategy

RerankingStrategy
└── BGERerankingStrategy
```

Problema real: el proyecto necesitaba comparar algoritmos intercambiables sin reescribir ingestion, retrieval ni evaluacion.

`EmbeddingStrategy` permitio comparar BGE-M3 y GTE Multilingual sobre el mismo corpus, los mismos chunks y el mismo ground truth. `RerankingStrategy` permitio comparar retrieval denso contra retrieval + CrossEncoder y luego dejar el reranker opcional.

### Factory Pattern

```text
LLMFactory
├── GroqLLMProvider
└── GeminiLLMProvider
```

Problema real: `RAGService` necesita generacion, pero no debe importar SDKs concretos. `LLMFactory` centraliza la seleccion del proveedor, modelo y credenciales. Cambiar `LLM_PROVIDER=groq` a `LLM_PROVIDER=gemini` no modifica la logica RAG.

## Technical Decisions

### BBVA -> Bancolombia

El objetivo inicial era BBVA Colombia. Durante el diagnostico, BBVA devolvia HTTP 403 con el stack Python estandar utilizado. El enunciado permitia usar otro banco, y Bancolombia funcionaba correctamente con `httpx` + BeautifulSoup para paginas publicas.

La decision fue usar Bancolombia para mantener scraping reproducible y evitar mecanismos de evasion como CAPTCHA bypass, proxies, fingerprint spoofing o curl impersonation. No se presenta como un fallo del proyecto, sino como una decision tecnica frente a restricciones reales.

### Raw vs Processed

Se conservan documentos raw y processed. Raw permite auditar y reprocesar sin volver a scrapear; processed contiene texto limpio para chunking, ingestion y evaluacion.

### Corpus Curado

El corpus se construyo desde sitemap y paginas de categoria, priorizando productos para personas. Se excluyeron home, formularios, login, PDFs, buscadores, duplicados y paginas de baja densidad informativa.

El corpus curado final tiene 40 `CleanDocuments` y 233 chunks. Se genero manifest reproducible con URL, titulo, categoria y `document_id`, y se aplico deduplicacion exacta por hash de contenido limpio.

### Chunking

`CHUNK_SIZE=900` y `CHUNK_OVERLAP=150` son un baseline simple, deterministico y medido en caracteres. Se eligieron para permitir experimentacion reproducible, no como valores universalmente optimos.

### Qdrant

Qdrant se eligio porque corre bien localmente con Docker, soporta cosine similarity, almacena payload metadata y puede migrar a Qdrant Cloud sin cambiar la logica de dominio. La coleccion final configurada es `bancolombia_eval_bge_m3`.

### BGE-M3 vs GTE

BGE-M3 fue seleccionado para este corpus por benchmark controlado, no porque sea universalmente superior a GTE.

Resultado BGE-M3:

- Hit@1: 0.6000
- Hit@5: 0.8000
- MRR: 0.6806

GTE Multilingual se conserva como implementacion y como alternativa experimental.

### Reranking Opcional

Dense retrieval:

```text
query -> embedding -> Qdrant -> Top K
```

Reranking:

```text
query -> embedding -> Qdrant Top candidates -> CrossEncoder -> reordered Top K
```

El CrossEncoder no se ejecuta contra todo el corpus porque evalua pares `(query, chunk)` y su costo crece linealmente con el numero de chunks. Primero Qdrant recupera candidatos con alto recall y luego el reranker reordena solo ese conjunto.

Resultados con BGE reranker:

- Hit@5: 0.9333
- MRR: 0.7844
- Candidate Recall@15: 0.9333

Trade-off: el reranker añadio aproximadamente 5.18 segundos de latencia CPU por query. Para un chatbot local interactivo, `RERANK_ENABLED=false` es el default. El reranker no fue eliminado; queda disponible por Strategy/configuracion cuando calidad tenga prioridad sobre latencia.

Los casos q015 y q026 eran problemas de ranking y fueron resueltos o mejorados por reranking. Los casos q009 y q013 eran problemas de candidate retrieval porque el chunk relevante no entro en Top 15, por lo que el reranker no podia recuperarlo.

### LLM + Grounding

Proveedor principal: Groq.

Modelo actual: `openai/gpt-oss-120b`.

Proveedor alternativo: Gemini.

El LLM devuelve una salida estructurada:

```json
{
  "answer": "...",
  "supported_by_context": true
}
```

Si `supported_by_context=false`, la API devuelve `sources=[]`. Esto evita mostrar fuentes irrelevantes solo porque Qdrant encontro vecinos semanticos para una pregunta fuera del dominio. Esta decision mejora la UX de grounding, pero no reemplaza una evaluacion formal de retrieval.

### SQLite

SQLite se eligio porque la prueba es local, requiere persistencia real, debe funcionar en Docker y no justifica PostgreSQL/Redis. La DB persiste en el volumen `conversation_data`. Para despliegues multi-replica o alta concurrencia, PostgreSQL seria una evolucion razonable.

### Torch CPU-only

La imagen API usa Torch CPU-only para evitar dependencias CUDA pesadas y mejorar reproducibilidad local. Esto prioriza facilidad de ejecucion sobre aceleracion GPU.

## Evaluation

### Offline Evaluation

La evaluacion offline compara retrieval y reranking sobre dataset curado y ground truth explicito. Estas metricas no deben mezclarse con analytics runtime.

Benchmark de embeddings:

```bash
uv --cache-dir .uv-cache run python -m scripts.evaluate_embeddings
```

El benchmark:

- usa los mismos documentos, chunks y `chunk_id` para ambos modelos
- crea colecciones Qdrant separadas
- mide Hit@1, Hit@3, Hit@5 y MRR
- separa carga de modelo, warmup, embedding de query y busqueda vectorial
- guarda resultados por pregunta y resumen machine-readable

Artefactos:

- `evaluation/retrieval_dataset.json`
- `evaluation/results_bge.json`
- `evaluation/results_gte.json`
- `evaluation/out_of_domain_results.json`
- `evaluation/summary.json`

Benchmark de reranker:

```bash
uv --cache-dir .uv-cache run python -m scripts.evaluate_reranker
```

Artefactos:

- `evaluation/results_reranker.json`
- `evaluation/reranker_summary.json`

Los scores de Qdrant y del CrossEncoder se preservan para debugging, pero no son directamente comparables porque viven en escalas distintas.

### Runtime Analytics

`GET /analytics/summary` calcula metricas recorriendo SQLite:

- `total_sessions`
- `total_messages`
- `total_user_messages`
- `total_assistant_messages`
- `supported_answers`
- `unsupported_answers`
- `supported_answer_rate`
- `average_total_latency_ms`
- `average_embedding_latency_ms`
- `average_search_latency_ms`
- `average_rerank_latency_ms`
- `average_llm_latency_ms`
- `average_sources_per_supported_answer`
- `average_messages_per_session`
- `average_user_messages_per_session`

`supported_answer_rate` no es accuracy. Es la proporcion de respuestas que el LLM marco como soportadas por el contexto recuperado.

Impact indicators incluidos:

- `supported_answer_rate`
- `average_response_latency_ms`
- `average_sources_per_supported_answer`

No se reportan customer satisfaction, time saved, productivity gain, ROI ni KPIs de negocio porque el sistema no recolecta datos para demostrar esas afirmaciones.

## Conversation Memory

La memoria conversacional se identifica por `session_id`.

- Si el cliente envia `session_id=null`, el backend crea un UUID.
- El frontend conserva el `session_id` para continuar la conversacion.
- El backend recupera los ultimos `CONVERSATION_HISTORY_N_MESSAGES=6` mensajes previos.
- Son mensajes, no turnos completos.
- El mensaje actual no se duplica en su propio prompt.
- El historial entra al `PromptBuilder`.
- Los mensajes user/assistant se persisten en SQLite al terminar exitosamente el turno.

El retrieval usa principalmente la pregunta actual. El historial ayuda al LLM a resolver referencias conversacionales en el prompt. Una mejora futura razonable seria conversational query rewriting para reformular follow-ups antes del vector search.

El frontend permite ver conversaciones anteriores cargando sesiones desde `GET /sessions` y mensajes desde `GET /sessions/{session_id}/messages`.

## Frontend

El frontend usa React, Vite y TypeScript.

```text
Browser -> /api/chat -> Vite proxy -> FastAPI
```

Tambien consume:

- `/api/sessions`
- `/api/sessions/{session_id}/messages`
- `/api/analytics/summary` disponible para inspeccion desde API

El frontend muestra conversacion, loading state, errores, fuentes clicables, estado unsupported y lista de conversaciones anteriores. Nunca recibe `GROQ_API_KEY` ni otras credenciales del backend.

## Docker

`docker-compose.yml` define:

- `qdrant`: vector database local.
- `api`: FastAPI + embeddings + RAG.
- `frontend`: React/Vite.

Volumenes:

- `qdrant_data`: persistencia de colecciones Qdrant.
- `conversation_data`: persistencia SQLite.
- `hf_cache`: cache de modelos Hugging Face.

La separacion de volumenes permite reiniciar containers con `docker compose down` y `docker compose up -d` sin perder vectores ni conversaciones, siempre que no se use `docker compose down -v`.

## Security

- `.env` esta en `.gitignore`.
- `.env.example` no contiene secretos.
- API keys existen solo en backend.
- El frontend no recibe `GROQ_API_KEY` ni `GEMINI_API_KEY`.
- Los errores HTTP no devuelven stack traces al navegador.
- `docker compose config` expande variables efectivas, por lo que no debe compartirse publicamente si `.env` contiene secretos.

## Useful Commands

Scraping + cleaning:

```bash
uv --cache-dir .uv-cache run python -m scripts.scrape_bancolombia
```

Preparar corpus curado:

```bash
uv --cache-dir .uv-cache run python -m scripts.prepare_corpus
```

Ingestion hacia Qdrant:

```bash
QDRANT_URL=http://localhost:6333 \
QDRANT_COLLECTION=bancolombia_eval_bge_m3 \
uv --cache-dir .uv-cache run python -m scripts.ingest
```

Consulta manual de retrieval:

```bash
QDRANT_URL=http://localhost:6333 \
QDRANT_COLLECTION=bancolombia_eval_bge_m3 \
EMBEDDING_PROVIDER=bge \
EMBEDDING_MODEL=BAAI/bge-m3 \
EMBEDDING_DEVICE=cpu \
uv --cache-dir .uv-cache run python -m scripts.retrieve \
  "¿Qué características tienen las tarjetas débito?" \
  --top-k 5
```

Inspeccionar coleccion Qdrant:

```bash
curl -s http://localhost:6333/collections/bancolombia_eval_bge_m3
```

## Tests

Validacion backend:

```bash
uv --cache-dir .uv-cache run --extra dev ruff check .
uv --cache-dir .uv-cache run --extra dev pytest
```

Resultado actual validado:

- `ruff`: All checks passed.
- `pytest`: 84 passed.

Validacion frontend:

```bash
cd frontend
npm run lint
npm run test
npm run build
```

Resultado actual validado:

- `npm run lint`: OK.
- `npm run test`: 6 passed.
- `npm run build`: OK.

Validacion Docker:

```bash
docker compose build
docker compose up -d
```

Resultado actual validado: build y stack local funcionando con `qdrant`, `api` y `frontend`.

## Limitations

- SQLite es adecuado para demo local/single-instance; multiples replicas o alta concurrencia requeririan PostgreSQL u otro store compartido.
- No hay autenticacion ni autorizacion multiusuario.
- Conversational query rewriting no esta implementado.
- `supported_by_context` no equivale a accuracy formal.
- Analytics runtime son indicadores tecnicos, no KPIs de negocio.
- `RAG_MIN_RETRIEVAL_SCORE` es una heuristica experimental y esta desactivada por defecto.
- La primera ejecucion puede descargar BGE-M3 si `hf_cache` esta vacio.
- El reranker mejora calidad pero tiene alta latencia en CPU.
- Torch CPU-only prioriza reproducibilidad local sobre aceleracion GPU.

## Future Improvements

- Conversational query rewriting para mejorar follow-ups ambiguos.
- PostgreSQL u otro store compartido para despliegues multi-replica.
- Autenticacion y autorizacion por usuario.
- Observability/tracing mas completo.
- Optimizacion o serving especializado del reranker.
- Evaluacion con dataset mas grande y mas categorias.
- Hybrid search o filtros por metadata si el corpus crece.
- Despliegue cloud en Google Cloud Run + Qdrant Cloud si fuera requerido.
