from pathlib import Path
from typing import Optional

import fitz


class DocumentScanner:
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def load(self, document: str, source_filename: Optional[str] = None) -> dict:
        return {
            "text": document,
            "source_filename": source_filename,
            "metadata": {
                "source": source_filename,
                "file_type": "text",
                "page_count": None,
                "total_characters": len(document),
            },
        }

    def load_document(self, file_path: str) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        file_type = path.suffix.lower()
        if file_type not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_type}")

        if file_type == ".pdf":
            return self._load_pdf(path)

        return self._load_text_file(path, file_type)

    def _load_pdf(self, path: Path) -> dict:
        pages = []
        with fitz.open(path) as document:
            for page_index, page in enumerate(document, start=1):
                page_text = page.get_text("text", sort=True)
                pages.append(
                    {
                        "page_number": page_index,
                        "text": page_text,
                        "character_count": len(page_text),
                    }
                )

        text = "\n\n".join(page["text"].strip() for page in pages if page["text"].strip())
        return self._document_result(
            text=text,
            source=path.name,
            file_type="pdf",
            page_count=len(pages),
            pages=pages,
        )

    def _load_text_file(self, path: Path, file_type: str) -> dict:
        text = path.read_text(encoding="utf-8", errors="replace")
        return self._document_result(
            text=text,
            source=path.name,
            file_type=file_type.lstrip("."),
            page_count=1,
            pages=[
                {
                    "page_number": 1,
                    "text": text,
                    "character_count": len(text),
                }
            ],
        )

    def _document_result(
        self,
        text: str,
        source: str,
        file_type: str,
        page_count: int,
        pages: list[dict],
    ) -> dict:
        return {
            "text": text,
            "source_filename": source,
            "metadata": {
                "source": source,
                "file_type": file_type,
                "page_count": page_count,
                "total_characters": len(text),
                "pages": pages,
            },
        }
