# Bancolombia Conversational RAG

Proyecto de prueba tecnica para construir un asistente conversacional RAG sobre informacion publica de `bancolombia.com`.

## Estado Actual

Phase 1 implementa la base arquitectonica:

- Estructura modular de paquetes.
- Configuracion centralizada por variables de entorno.
- Contratos base para Repository, Strategy y Factory.
- Modelos Pydantic compartidos.
- Endpoint minimo `GET /health`.

Phase 2 implementa scraping y cleaning:

- Scraper especifico para Bancolombia.
- Frontera de dominio configurable.
- Limite de paginas, timeout y rate limit.
- Filtros derivados de `robots.txt`: formularios, buscadores, PDFs, rutas REST/CGI, solicitudes de productos, preaprobados y ofertas.
- Extraccion de URL, titulo, contenido, seccion y fecha de scraping.
- Cleaner separado para normalizar texto y reducir ruido repetitivo.
- Persistencia JSONL para `raw` y `processed`.

Phase 3 implementa chunking, embeddings, Qdrant e ingestion:

- Chunking determinista sobre documentos procesados.
- `CHUNK_SIZE` y `CHUNK_OVERLAP` medidos en caracteres.
- Embeddings locales mediante `sentence-transformers`.
- Estrategias configurables para BGE-M3 y GTE Multilingual Base.
- Repositorio vectorial Qdrant detrás de `VectorRepository`.
- Pipeline explicito de ingestion desde JSONL procesado hasta Qdrant.

Phase 4 implementa retrieval funcional:

- `Retriever` independiente que depende solo de `EmbeddingStrategy` y `VectorRepository`.
- Embedding de la pregunta con el modelo configurado.
- Busqueda vectorial Top K en Qdrant usando cosine similarity.
- CLI de debug para inspeccionar chunks recuperados, scores y tiempos.

Phase 7 implementa el primer RAG completo local:

- `POST /chat` recibe una pregunta y devuelve respuesta + fuentes.
- `RAGService` orquesta retrieval, contexto, prompt y LLM.
- `ContextBuilder` transforma chunks recuperados en contexto estructurado.
- `PromptBuilder` aplica grounding: responder solo con contexto y rechazar evidencia insuficiente.
- `LLMFactory` selecciona Groq o Gemini por configuracion.
- El reranker queda disponible pero desactivado por defecto con `RERANK_ENABLED=false`.

Phase 8 productiza la ejecucion local:

- Salida estructurada del LLM con `supported_by_context`.
- Supresion de fuentes cuando la respuesta no esta soportada por el contexto.
- API dockerizada con Python 3.12 y Torch CPU-only.
- Frontend React/Vite minimo para probar el flujo.
- Docker Compose con `qdrant`, `api` y `frontend`.
- Volumen persistente para Qdrant y cache de Hugging Face.

Preparacion de corpus para evaluacion:

- Descubrimiento desde `sitemap-personas.xml`.
- Seleccion curada de paginas de productos para personas.
- Exclusion de home porque funciona mejor como navegacion/descubrimiento que como fuente densa de conocimiento.
- Manifest reproducible con URL, titulo, categoria y `document_id`.
- Reporte de calidad con descartes, duplicados exactos, estadisticas de contenido y estadisticas de chunks.

Memoria conversacional avanzada y despliegue cloud quedan fuera de esta fase.

## Arquitectura Objetivo

Ingestion pipeline:

```text
Bancolombia -> Scraper -> Raw Data -> Cleaner -> Processed Data -> Chunker
     -> Embedding Strategy -> Vector Repository -> Qdrant
```

Query pipeline:

```text
User -> Frontend -> API -> RAGService -> RetrievalPipeline -> Retriever -> Qdrant
     -> [optional Reranker] -> ContextBuilder -> PromptBuilder -> LLM
     -> Response + Sources
```

Estado de Phase 7 con `RERANK_ENABLED=false`:

```text
Question -> BGE-M3 -> Qdrant cosine search -> Top 5 chunks
         -> ContextBuilder -> PromptBuilder -> LLM -> grounded answer
```

Estado de Phase 7 con `RERANK_ENABLED=true`:

```text
Question -> BGE-M3 -> Qdrant Top 15 candidates
         -> BGE reranker -> Top 5 chunks -> ContextBuilder
         -> PromptBuilder -> LLM -> grounded answer
```

Retrieval no es generation: retrieval recupera fragmentos similares, mientras que generation redacta una respuesta usando solamente ese contexto. La suficiencia del contexto se controla con instrucciones de grounding y, opcionalmente, con `RAG_MIN_RETRIEVAL_SCORE` como heuristica experimental desactivada por defecto.

Si el LLM devuelve `supported_by_context=false`, la API responde con `sources=[]`. Esta es una decision de UX/grounding para no mostrar fuentes irrelevantes cuando Qdrant devuelve vecinos semanticos pero no evidencia suficiente.

## Design Patterns

Repository Pattern:

```text
VectorRepository
      ↑
QdrantVectorRepository
```

Problema real: ingestion y retrieval necesitan guardar y buscar vectores, pero no deberian conocer `qdrant-client`. El contrato permite mantener `RAGService`, `Retriever` e `IngestionPipeline` desacoplados de Qdrant. Ejemplo concreto: la API puede apuntar a `http://qdrant:6333` local o a Qdrant Cloud cambiando variables de entorno, sin cambiar logica RAG.

Strategy Pattern:

```text
EmbeddingStrategy
├── BGEM3EmbeddingStrategy
└── GTEMultilingualEmbeddingStrategy

RerankingStrategy
└── BGERerankingStrategy
```

Problema real: queriamos comparar embeddings y rerankers sin reescribir ingestion, retrieval ni evaluacion. `EmbeddingStrategy` permitio evaluar BGE-M3 vs GTE sobre los mismos chunks. `RerankingStrategy` permitio medir calidad vs latencia y dejar el reranker opcional con `RERANK_ENABLED=false`.

Factory Pattern:

```text
LLMFactory
├── GroqLLMProvider
└── GeminiLLMProvider
```

Problema real: `RAGService` debe generar respuestas, pero no debe depender del SDK de Groq o Gemini. La Factory centraliza seleccion, credenciales y modelo configurado. Ejemplo concreto: `LLM_PROVIDER=groq` o `LLM_PROVIDER=gemini` cambia el proveedor sin modificar `RAGService`.

## Architecture Decisions

BBVA -> Bancolombia:
BBVA Colombia devolvia HTTP 403 con clientes Python estandar en el entorno de desarrollo. Bancolombia permitio scraping HTTP estandar respetando `robots.txt`, limites y rate limiting, por lo que se adopto como fuente alternativa permitida por el enunciado.

Raw vs processed:
Se guardan documentos raw y processed para poder reprocesar, limpiar o rechunkear sin volver a descargar paginas publicas. Esto reduce variabilidad y hace reproducibles las evaluaciones.

Cleaner estructural:
El cleaner separa extraccion de limpieza y reduce navegacion, footer, cookies y texto repetitivo. La decision mantiene el scraper simple y evita mezclar responsabilidades.

Chunking 900/150:
`CHUNK_SIZE=900` y `CHUNK_OVERLAP=150` son un baseline simple, determinista y medido en caracteres. No se presentan como optimos; se eligieron para experimentar de forma reproducible.

Qdrant mediante Repository:
`VectorRepository` aisla Qdrant de ingestion, retrieval y RAG. Cambiar a Qdrant Cloud o a otra base vectorial no deberia modificar la logica del dominio.

BGE-M3 vs GTE:
BGE-M3 fue seleccionado para el pipeline por resultados de nuestro benchmark, no por superioridad universal. Resultado BGE-M3: Hit@1 0.6000, Hit@5 0.8000, MRR 0.6806. GTE se conserva para experimentacion y evaluaciones futuras.

Reranker opcional:
Reranking improved retrieval quality substantially, but added ~5.18 s of CPU latency per query. It is therefore retained as an optional strategy but disabled by default for interactive CPU deployment.

En Phase 6, dense + reranker mejoro Hit@5 a 0.9333 y MRR a 0.7844, con Candidate Recall@15 de 0.9333. Los casos q015 y q026 eran problemas de ranking y fueron resueltos o mejorados por reranking. Los casos q009 y q013 eran problemas de candidate retrieval porque el relevante no entro en Top 15, por lo que el reranker no podia recuperarlo.

LLM mediante Factory:
Groq es el proveedor principal configurado con `openai/gpt-oss-120b`; Gemini queda como alternativa. La Factory centraliza la seleccion de proveedor para que `RAGService` no conozca SDKs concretos.

## Configuracion

Copiar `.env.example` a `.env` y ajustar valores locales.

Variables relevantes de Phase 3:

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

LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=
GEMINI_API_KEY=
```

`CHUNK_SIZE` y `CHUNK_OVERLAP` usan caracteres, no tokens. Los valores iniciales son deliberadamente configurables para experimentar despues.

## Ejecucion Local

### Docker End-to-End

Prerequisites:

- Docker
- Docker Compose v2

Setup desde una maquina limpia:

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

Ejecutar ingestion offline dentro del contenedor API:

```bash
docker compose run --rm api python -m scripts.ingest
```

Esto usa `data/corpus/processed_documents.jsonl`, genera chunks, calcula embeddings BGE-M3 y hace upsert en Qdrant. No ejecuta scraping automatico.

Levantar stack completo:

```bash
docker compose up --build
```

URLs locales:

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Qdrant: `http://localhost:6333`

Qdrant persiste en el volumen `qdrant_data`. La cache de modelos se reutiliza en el volumen `hf_cache`; si BGE-M3 no existe en cache, se descarga en la primera ingestion o primera query que cargue embeddings.

### Ejecucion Host Con uv

Si ejecutas scripts desde el host con `uv`, usa:

```bash
QDRANT_URL=http://localhost:6333 \
QDRANT_COLLECTION=bancolombia_eval_bge_m3 \
RERANK_ENABLED=false \
uv --cache-dir .uv-cache run uvicorn app.main:app --reload
```

Si la aplicacion corre dentro de Docker Compose, usa `QDRANT_URL=http://qdrant:6333`.

Request:

```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"¿Qué opciones ofrece Bancolombia para financiar vivienda?"}'
```

Response:

```json
{
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

Para ejecutar scraping + cleaning:

```bash
uv --cache-dir .uv-cache run python -m scripts.scrape_bancolombia
```

El script guarda documentos en:

- `data/raw/bancolombia_documents.jsonl`
- `data/processed/bancolombia_documents.jsonl`

Estos archivos no se versionan porque dependen de una ejecucion real contra el sitio publico.

Para ejecutar ingestion hacia Qdrant:

```bash
uv --cache-dir .uv-cache run python -m scripts.ingest
```

El comando carga `data/processed/bancolombia_documents.jsonl`, genera chunks, calcula embeddings locales y hace upsert en la coleccion configurada.

Para inspeccionar la coleccion desde Python:

```bash
uv --cache-dir .uv-cache run python - <<'PY'
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
collection = "bancolombia_bge"
print(client.get_collection(collection))
print(client.count(collection_name=collection, exact=True))
PY
```

Para ejecutar una consulta manual de retrieval:

```bash
QDRANT_URL=http://localhost:6333 \
QDRANT_COLLECTION=bancolombia_bge_phase3_test \
EMBEDDING_PROVIDER=bge \
EMBEDDING_MODEL=BAAI/bge-m3 \
EMBEDDING_DEVICE=cpu \
uv --cache-dir .uv-cache run python -m scripts.retrieve \
  "¿Qué características tienen las tarjetas débito?" \
  --top-k 5
```

El comando muestra:

- pregunta
- modelo de embedding
- coleccion Qdrant
- tiempo aproximado de embedding de query
- tiempo aproximado de busqueda vectorial
- Top K chunks con score, titulo, URL, indice y texto

No se aplica threshold en Phase 4. Qdrant siempre puede devolver los vectores mas similares aunque una pregunta este fuera del dominio del corpus; esa validacion de suficiencia se resolvera en fases posteriores.

## Frontend

El frontend esta en `frontend/` y usa React + Vite. Consume `POST /chat` mediante el proxy `/api` de Vite:

```text
Browser -> http://localhost:3000/api/chat -> Vite proxy -> http://api:8000/chat
```

Esto evita exponer URLs internas de Docker al navegador y mantiene `GROQ_API_KEY` exclusivamente en el backend. El frontend muestra:

- conversacion
- estado de carga
- errores de API
- respuestas
- fuentes clicables cuando `supported_by_context=true`
- indicador discreto sin fuentes cuando `supported_by_context=false`

## Offline vs Online

Offline ingestion:

```text
Bancolombia -> Scraper -> Raw -> Cleaner -> CleanDocuments -> Chunker -> BGE -> Qdrant
```

Online inference:

```text
User -> Frontend -> POST /chat -> RAGService -> BGE query embedding
     -> Qdrant -> optional reranker -> ContextBuilder -> LLM -> answer
```

El scraping no ocurre en cada pregunta. La coleccion Qdrant debe existir antes de usar el chatbot en una maquina limpia.

## Seguridad Basica

- `.env` esta en `.gitignore`.
- API keys no se copian dentro de la imagen.
- API keys no se envian al frontend.
- El frontend solo llama al backend propio mediante `/api`.
- Los errores HTTP devueltos al navegador no incluyen stack traces.
- `docker compose config` muestra variables efectivas, por lo que no debe compartirse publicamente si `.env` contiene secretos.

Para preparar el corpus curado de evaluacion:

```bash
uv --cache-dir .uv-cache run python -m scripts.prepare_corpus
```

El comando genera:

- `data/corpus/raw_documents.jsonl`
- `data/corpus/processed_documents.jsonl`
- `data/corpus/manifest.json`
- `data/corpus/quality_report.json`

Este paso no indexa en Qdrant ni sobrescribe colecciones existentes. Phase 5 creara colecciones nuevas para comparar embeddings sobre este mismo corpus.

## Tests

```bash
pytest
```

## Evaluation

Phase 5 compara retrieval denso BGE-M3 vs GTE Multilingual Base sobre el corpus congelado de `data/corpus/processed_documents.jsonl`.

```bash
uv --cache-dir .uv-cache run python -m scripts.evaluate_embeddings
```

El benchmark:

- usa los mismos documentos, chunks y `chunk_id` para ambos modelos
- crea colecciones Qdrant separadas
- mide Hit@1, Hit@3, Hit@5 y MRR
- separa tiempo de carga, warmup, embedding de query y busqueda vectorial
- guarda resultados por pregunta y resumen machine-readable

Artefactos:

- `evaluation/retrieval_dataset.json`
- `evaluation/results_bge.json`
- `evaluation/results_gte.json`
- `evaluation/out_of_domain_results.json`
- `evaluation/summary.json`

Nota reproducible: `Alibaba-NLP/gte-multilingual-base` requiere `trust_remote_code=True` y fallo con `transformers>=5` por un problema de `position_ids` reportado en Hugging Face. El proyecto pinnea `sentence-transformers==3.4.1` y `transformers==4.53.3` para ejecutar BGE y GTE localmente de forma estable.

Resultado usado para seleccionar BGE-M3 en este corpus:

- Hit@1: 0.6000
- Hit@5: 0.8000
- MRR: 0.6806

Phase 6 evalua reranking sobre BGE-M3:

```text
Dense retrieval:
query -> BGE-M3 -> Qdrant cosine search -> Top K

Reranking:
query -> BGE-M3 -> Qdrant Top candidates -> CrossEncoder -> reordered Top K
```

El CrossEncoder no se ejecuta contra todo el corpus porque evalua pares `(query, chunk)` y su costo crece linealmente con el numero de chunks. Primero se usa Qdrant para recuperar candidatos con alto recall y luego el reranker reordena solo ese conjunto pequeno.

```bash
uv --cache-dir .uv-cache run python -m scripts.evaluate_reranker
```

Artefactos:

- `evaluation/results_reranker.json`
- `evaluation/reranker_summary.json`

Los scores de Qdrant y del CrossEncoder se guardan para debugging, pero no son directamente comparables porque viven en escalas distintas.

Decision para el pipeline final local:

```bash
RERANK_ENABLED=false
```

El reranker permanece disponible para escenarios donde la mejora de calidad compense la latencia extra.

## Limitaciones Actuales

- No hay memoria conversacional avanzada; Phase 8 valida el RAG local de una pregunta.
- El threshold `RAG_MIN_RETRIEVAL_SCORE` existe solo como heuristica experimental y no esta activado por defecto.
- El primer arranque puede descargar BGE-M3 si el volumen `hf_cache` esta vacio.
- La imagen API usa Torch CPU-only para evitar dependencias CUDA pesadas en despliegue local.
