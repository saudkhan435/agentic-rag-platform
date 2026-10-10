from io import BytesIO

import fitz
import pytesseract
from PIL import Image


def extract_page_with_ocr(
    file_path: str,
    page_index: int
) -> str:

    with fitz.open(file_path) as document:
        page = document.load_page(page_index)

        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(3, 3),
            alpha=False
        )

        image = Image.open(
            BytesIO(
                pixmap.tobytes("png")
            )
        )

        text = pytesseract.image_to_string(image)

        return text.strip()