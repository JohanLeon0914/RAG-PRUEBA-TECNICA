from datetime import UTC, datetime

from app.schemas import SourceDocument
from app.scraping.cleaner import BankContentCleaner


def test_cleaner_normalizes_text_and_removes_consecutive_duplicates() -> None:
    cleaner = BankContentCleaner()

    cleaned = cleaner.clean_text(
        """
        Aceptar cookies

        Tarjeta Aqua
        Tarjeta Aqua
        Sin cuota de manejo durante los primeros meses.   


        Configurar cookies
        """
    )

    assert cleaned == "Tarjeta Aqua\nSin cuota de manejo durante los primeros meses."


def test_cleaner_removes_ui_tokens_buttons_and_repeated_lines() -> None:
    cleaned = BankContentCleaner().clean_text(
        """
        Inicio
        arrow2-right
        Tarjetas de crédito
        plus
        Ahorra 5 meses de cuota pidiéndola online.
        Conoce más
        Ahorra 5 meses de cuota pidiéndola online.
        hand-holding-cash
        Cuota de manejo variable
        Según el plan de tu cuenta
        Sucursal Virtual Personas
        """
    )

    assert cleaned == (
        "Tarjetas de crédito\n"
        "Ahorra 5 meses de cuota pidiéndola online.\n"
        "Cuota de manejo variable\n"
        "Según el plan de tu cuenta"
    )


def test_cleaner_removes_global_header_and_footer_blocks() -> None:
    cleaned = BankContentCleaner().clean_text(
        """
        {}
        Personas
        Negocios
        Bancolombia
        Banco Agrícola
        BAM (Banco Agromercantil de Guatemala)
        Crédito de Libre Inversión
        Estudia, remodela, emprende y más.
        Beneficios para ti
        Tasa preferencial si adquieres seguro.
        Te puede interesar
        Accesibilidad
        Contáctanos
        Carrera 48 # 26 - 85 Medellín – Colombia
        Copyright © 2026 Bancolombia
        Deferred Modules
        ${badge}
        """
    )

    assert cleaned == (
        "Crédito de Libre Inversión\n"
        "Estudia, remodela, emprende y más.\n"
        "Beneficios para ti\n"
        "Tasa preferencial si adquieres seguro."
    )


def test_cleaner_removes_global_header_when_bam_marker_is_split() -> None:
    cleaned = BankContentCleaner().clean_text(
        """
        {}
        Personas
        Negocios
        BAM
        (Banco Agromercantil
        de
        Guatemala)
        Créditos, hipotecas y leasing de Vivienda Bancolombia
        Compra o remodela vivienda con diferentes opciones de financiación.
        """
    )

    assert cleaned == (
        "Créditos, hipotecas y leasing de Vivienda Bancolombia\n"
        "Compra o remodela vivienda con diferentes opciones de financiación."
    )


def test_cleaner_preserves_document_metadata() -> None:
    scraped_at = datetime.now(UTC)
    document = SourceDocument(
        document_id="doc-1",
        url="https://www.bancolombia.com/personas/productos/tarjetas-credito",
        title="Tarjetas",
        section="personas",
        content="Tarjetas\nTarjetas\nBeneficios",
        scraped_at=scraped_at,
        metadata={"source": "bancolombia.com"},
    )

    cleaned = BankContentCleaner().clean([document])

    assert cleaned[0].document_id == "doc-1"
    assert str(cleaned[0].url) == (
        "https://www.bancolombia.com/personas/productos/tarjetas-credito"
    )
    assert cleaned[0].scraped_at == scraped_at
    assert cleaned[0].metadata == {"source": "bancolombia.com"}
    assert cleaned[0].content == "Tarjetas\nBeneficios"
