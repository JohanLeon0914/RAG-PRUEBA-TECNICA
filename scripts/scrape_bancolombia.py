from app.config.settings import get_settings
from app.scraping.cleaner import BankContentCleaner
from app.scraping.scraper import BancolombiaScraper
from app.scraping.storage import save_documents_jsonl


def main() -> None:
    settings = get_settings()
    scraper = BancolombiaScraper.from_settings(settings)
    cleaner = BankContentCleaner()

    raw_documents = scraper.scrape()
    save_documents_jsonl(raw_documents, settings.raw_data_path)

    processed_documents = cleaner.clean(raw_documents)
    save_documents_jsonl(processed_documents, settings.processed_data_path)

    print(
        f"Scraped {len(raw_documents)} raw documents and "
        f"saved {len(processed_documents)} processed documents."
    )


if __name__ == "__main__":
    main()
