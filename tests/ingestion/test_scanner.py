import pytest

from src.ingestion.scanner import DocumentScanner


def test_scanner_loads_pdf_with_page_metadata(tmp_path, write_pdf):
    pdf_path = tmp_path / "nexon-brochure.pdf"
    write_pdf(
        pdf_path,
        [
            "Safety Section\nAdvanced safety features including 6 airbags.",
            "Performance Section\nPowerful 1.2L turbo engine with great mileage.",
        ],
    )

    loaded_document = DocumentScanner().load_document(str(pdf_path))

    assert "Safety Section" in loaded_document["text"]
    assert "Performance Section" in loaded_document["text"]
    assert loaded_document["metadata"]["source"] == "nexon-brochure.pdf"
    assert loaded_document["metadata"]["file_type"] == "pdf"
    assert loaded_document["metadata"]["page_count"] == 2
    assert loaded_document["metadata"]["total_characters"] == len(loaded_document["text"])
    assert loaded_document["metadata"]["pages"][0]["page_number"] == 1


def test_scanner_rejects_unsupported_file_type(tmp_path):
    unsupported_file = tmp_path / "nexon.xlsx"
    unsupported_file.write_text("not supported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentScanner().load_document(str(unsupported_file))
