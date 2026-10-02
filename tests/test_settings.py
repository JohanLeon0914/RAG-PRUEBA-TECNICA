from app.config.settings import Settings


def test_settings_parse_start_urls_from_comma_separated_env_value() -> None:
    settings = Settings(
        BANK_START_URLS=(
            "https://www.bancolombia.com/personas, "
            "https://www.bancolombia.com/personas/tarjetas-de-credito"
        )
    )

    assert settings.bank_start_urls == [
        "https://www.bancolombia.com/personas",
        "https://www.bancolombia.com/personas/tarjetas-de-credito",
    ]


def test_settings_parse_cors_origins_from_comma_separated_env_value() -> None:
    settings = Settings(
        CORS_ALLOWED_ORIGINS="http://localhost:3000, http://127.0.0.1:3000"
    )

    assert settings.cors_allowed_origins == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


def test_settings_use_bancolombia_seeds_by_default() -> None:
    settings = Settings()

    assert settings.bank_start_urls[0] == "https://www.bancolombia.com/"
    assert settings.bank_allowed_domain == "bancolombia.com"
    assert settings.scraper_http2 is True


def test_settings_reject_chunk_overlap_equal_to_chunk_size() -> None:
    try:
        Settings(CHUNK_SIZE=100, CHUNK_OVERLAP=100)
    except ValueError as exc:
        assert "CHUNK_OVERLAP" in str(exc)
    else:
        raise AssertionError("Settings should reject chunk_overlap >= chunk_size")


def test_settings_normalize_empty_secret_values() -> None:
    settings = Settings(QDRANT_API_KEY="", GROQ_API_KEY="", GEMINI_API_KEY="")

    assert settings.qdrant_api_key is None
    assert settings.groq_api_key is None
    assert settings.gemini_api_key is None
