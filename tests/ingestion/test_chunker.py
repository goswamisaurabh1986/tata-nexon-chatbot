from src.ingestion.chunker import DocumentChunker


def test_chunking_respects_section_boundaries_and_preserves_section_titles(section_document):
    chunks = DocumentChunker().chunk(section_document, source_filename="nexon-brochure.txt")

    assert len(chunks) >= 2
    section_titles = [chunk["metadata"]["section_title"] for chunk in chunks]
    assert "Safety Section" in section_titles
    assert "Performance Section" in section_titles
    assert [chunk["metadata"]["section"] for chunk in chunks] == [
        "Safety Section",
        "Performance Section",
    ]


def test_long_document_chunking_respects_chunk_size_and_overlap(long_document):
    chunks = DocumentChunker().chunk(
        long_document,
        source_filename="nexon-brochure.txt",
        chunk_size=200,
        overlap=50,
    )

    assert len(chunks) > 1
    chunk_texts = [chunk["text"] for chunk in chunks]
    assert all(len(text) <= 200 for text in chunk_texts)
    for previous, current in zip(chunk_texts, chunk_texts[1:]):
        assert previous[-50:] in current


def test_each_section_chunk_contains_rich_metadata(section_document):
    chunks = DocumentChunker().chunk(
        section_document,
        source_filename="nexon-brochure.txt",
        chunk_size=200,
    )

    assert len(chunks) >= 2
    section_titles = [chunk["metadata"]["section_title"] for chunk in chunks]
    assert "Safety Section" in section_titles
    assert "Performance Section" in section_titles
    for index, chunk in enumerate(chunks):
        metadata = chunk["metadata"]
        assert metadata["source"] == "nexon-brochure.txt"
        assert metadata["chunk_index"] == index
        assert metadata["total_chunks"] == len(chunks)
        assert metadata["chunk_size"] == 200
