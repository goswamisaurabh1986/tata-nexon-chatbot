from src.ingestion.storer import VectorStorer


def test_storer_receives_chunks_with_embeddings_and_metadata(
    embedded_chunks,
    mock_storer,
):
    VectorStorer(vector_store=mock_storer).store(
        embedded_chunks,
        source_filename="nexon-brochure.txt",
        document="Tata Nexon Features",
    )

    mock_storer.store.assert_called_once_with(embedded_chunks)
    stored_chunks = mock_storer.store.call_args.args[0]
    assert len(stored_chunks) == 2
    for chunk in stored_chunks:
        assert chunk["embedding"]
        assert chunk["metadata"]["source"]
        assert chunk["metadata"]["section_title"]


def test_processing_same_document_twice_does_not_store_duplicates(
    embedded_chunks,
    mock_storer,
):
    storer = VectorStorer(vector_store=mock_storer)

    first_chunks = storer.store(
        embedded_chunks,
        source_filename="nexon-brochure.txt",
        document="Tata Nexon Features",
    )
    second_chunks = storer.store(
        embedded_chunks,
        source_filename="nexon-brochure.txt",
        document="Tata Nexon Features",
    )

    assert first_chunks == second_chunks
    mock_storer.store.assert_called_once_with(first_chunks)


def test_vector_storer_persists_metadata_and_avoids_duplicates(
    tmp_path,
    embedded_chunks,
):
    storer = VectorStorer(
        collection_name="test_nexon_chunks",
        persist_directory=str(tmp_path / "chroma"),
        embedding_dimension=3,
    )

    storer.store(embedded_chunks)
    storer.store(embedded_chunks)

    assert storer.get_collection_name() == "test_nexon_chunks"
    assert storer.collection.count() == 2
    stored = storer.collection.get(include=["documents", "metadatas"])
    assert stored["documents"] == [chunk["text"] for chunk in embedded_chunks]
    assert stored["metadatas"][0]["source"] == "nexon-brochure.pdf"
    assert stored["metadatas"][1]["section_title"] == "Performance Section"
