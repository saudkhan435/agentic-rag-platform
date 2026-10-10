from pathlib import Path

from pypdf import PdfReader

from app.ingestion.ocr_loader import extract_page_with_ocr


OCR_TEXT_THRESHOLD = 300


def extract_text_from_pdf(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    reader = PdfReader(path)

    pages = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        if len(text.strip()) < OCR_TEXT_THRESHOLD:
            ocr_text = extract_page_with_ocr(
                file_path=file_path,
                page_index=page_index
            )

            if len(ocr_text) > len(text):
                text = ocr_text

        if text.strip():
            pages.append(text)

    return "\n".join(pages)