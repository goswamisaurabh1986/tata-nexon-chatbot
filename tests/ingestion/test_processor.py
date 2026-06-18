import pytest

from src.ingestion.processor import IngestionProcessor, process_document


@pytest.mark.parametrize("document", ["", "   \n\t  "])
def test_empty_or_whitespace_document_returns_empty_chunks(document):
    assert process_document(document) == []


def test_single_sentence_document_returns_one_chunk_with_metadata():
    document = "The Tata Nexon offers advanced safety features."

    chunks = process_document(document, source_filename="nexon-brochure.txt")

    assert chunks == [
        {
            "text": document,
            "metadata": {
                "source": "nexon-brochure.txt",
                "chunk_index": 0,
            },
        }
    ]


def test_processor_uses_embed_batch_for_chunk_embeddings(
    section_document,
    mock_batch_embedder,
):
    chunks = process_document(
        section_document,
        source_filename="nexon-brochure.txt",
        embedder=mock_batch_embedder,
    )

    mock_batch_embedder.embed_batch.assert_called_once_with(
        [chunk["text"] for chunk in chunks]
    )
    assert [chunk["embedding"] for chunk in chunks] == [[0.1, 0.2], [0.3, 0.4]]


def test_processor_chunks_multi_page_pdf_and_preserves_file_metadata(
    tmp_path,
    write_pdf,
):
    pdf_path = tmp_path / "nexon-brochure.pdf"
    write_pdf(
        pdf_path,
        [
            "Safety Section\nAdvanced safety features including 6 airbags.",
            "Performance Section\nPowerful 1.2L turbo engine with great mileage.",
        ],
    )

    chunks = IngestionProcessor().process(file_path=str(pdf_path), chunk_size=120, overlap=20)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk["metadata"]["source"] == "nexon-brochure.pdf"
        assert chunk["metadata"]["file_type"] == "pdf"
        assert chunk["metadata"]["page_count"] == 2
