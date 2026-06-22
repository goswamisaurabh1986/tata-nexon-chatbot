from unittest.mock import Mock

from src.ingestion.embedder import Embedder


def test_embedder_adds_embedding_to_each_chunk(mock_embedder):
    chunks = [
        {"text": "Advanced safety features including 6 airbags.", "metadata": {}},
        {"text": "Powerful 1.2L turbo engine with great mileage.", "metadata": {}},
    ]

    embedded_chunks = Embedder(embedding_model=mock_embedder).embed_chunks(chunks)

    assert mock_embedder.embed.call_count == 2
    for chunk in embedded_chunks:
        assert chunk["embedding"] == [0.1, 0.2, 0.3]
        assert all(isinstance(value, float) for value in chunk["embedding"])


def test_embedder_skips_empty_text_in_batch():
    client = Mock()
    client.embeddings.create.return_value.data = [Mock(embedding=[0.1, 0.2, 0.3])]
    embedder = Embedder(api_key="test-key", client=client)

    embeddings = embedder.embed_batch(["", "   ", "Useful Tata Nexon content"])

    assert embeddings == [[], [], [0.1, 0.2, 0.3]]
    client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["Useful Tata Nexon content"],
    )


def test_embedder_batches_large_requests():
    client = Mock()
    client.embeddings.create.side_effect = [
        Mock(data=[Mock(embedding=[0.1]), Mock(embedding=[0.2])]),
        Mock(data=[Mock(embedding=[0.3])]),
    ]
    embedder = Embedder(api_key="test-key", client=client, batch_size=2)

    embeddings = embedder.embed_batch(["one", "two", "three"])

    assert embeddings == [[0.1], [0.2], [0.3]]
    assert client.embeddings.create.call_count == 2
    assert client.embeddings.create.call_args_list[0].kwargs["input"] == ["one", "two"]
    assert client.embeddings.create.call_args_list[1].kwargs["input"] == ["three"]
