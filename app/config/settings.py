from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Bancolombia RAG"
    app_env: str = "local"
    log_level: str = "INFO"
    cors_allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "bancolombia_eval_bge_m3"

    embedding_provider: Literal["bge", "gte"] = "bge"
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 16
    embedding_device: Literal["cpu", "cuda"] = "cpu"

    reranker_provider: Literal["bge", "gte"] = "bge"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: Literal["cpu", "cuda"] = "cpu"
    reranker_batch_size: int = 16

    llm_provider: Literal["groq", "gemini"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    llm_timeout_seconds: int = 30
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800

    bank_start_urls: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "https://www.bancolombia.com/",
            "https://www.bancolombia.com/personas/tarjetas-de-credito",
            "https://www.bancolombia.com/personas/productos/cuentas/tarjetas-debito",
        ]
    )
    bank_allowed_domain: str = "bancolombia.com"
    scraper_user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    )
    scraper_accept: str = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    scraper_accept_language: str = "es-CO,es;q=0.9,en;q=0.8"
    scraper_accept_encoding: str = "gzip, deflate"
    scraper_http2: bool = True
    scraper_max_pages: int = 30
    scraper_timeout_seconds: int = 15
    scraper_rate_limit_seconds: float = 1.0
    scraper_disallowed_patterns: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "/*myportal*",
            "/rest/*",
            "/cgi/*",
            "/crypto-js/*",
            "/*formulario*",
            "/*?ofertaId*",
            "/*pdf*",
            "/*!ut/p/z0/*",
            "/*!ut/p/z1/*",
            "/*buscador*",
            "/wps/portal/preguntas-frecuentes/preguntas/*",
            "/personas/solicitud-de-productos/*",
            "*/preaprobados*",
            "*/preaprobados-digitales*",
            "*/oferta/*",
            "/centro-de-ayuda/preguntas-frecuentes/resultados*",
        ]
    )
    raw_data_path: str = "data/raw/bancolombia_documents.jsonl"
    processed_data_path: str = "data/processed/bancolombia_documents.jsonl"

    chunk_size: int = 900
    chunk_overlap: int = 150

    retrieval_top_k: int = 10
    retrieval_candidate_k: int = 15
    rerank_enabled: bool = False
    rerank_top_k: int = 5
    rerank_top_n: int = 4
    rag_context_top_k: int = 5
    rag_min_retrieval_score: float | None = None
    memory_max_messages: int = 8

    @field_validator("bank_start_urls", mode="before")
    @classmethod
    def parse_start_urls(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [url.strip() for url in value.split(",") if url.strip()]
        return value

    @field_validator("scraper_disallowed_patterns", mode="before")
    @classmethod
    def parse_disallowed_patterns(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [pattern.strip() for pattern in value.split(",") if pattern.strip()]
        return value

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def parse_cors_allowed_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be lower than CHUNK_SIZE")
        return value

    @field_validator(
        "scraper_max_pages",
        "scraper_timeout_seconds",
        "chunk_size",
        "embedding_batch_size",
        "reranker_batch_size",
        "retrieval_top_k",
        "retrieval_candidate_k",
        "rerank_top_k",
        "rerank_top_n",
        "rag_context_top_k",
        "llm_timeout_seconds",
        "llm_max_tokens",
        "memory_max_messages",
    )
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("scraper_rate_limit_seconds")
    @classmethod
    def validate_non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("value must be non-negative")
        return value

    @field_validator("qdrant_api_key", "groq_api_key", "gemini_api_key", mode="before")
    @classmethod
    def empty_secret_to_none(cls, value: str | None) -> str | None:
        if value == "":
            return None
        return value

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if value < 0:
            raise ValueError("LLM_TEMPERATURE must be non-negative")
        return value

    @field_validator("rag_min_retrieval_score", mode="before")
    @classmethod
    def validate_optional_score(cls, value: float | str | None) -> float | None:
        if value == "":
            return None
        if value is not None:
            value = float(value)
        if value is not None and not 0 <= value <= 1:
            raise ValueError("RAG_MIN_RETRIEVAL_SCORE must be between 0 and 1")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
