from unittest.mock import Mock

import pytest

from src.retrieval.retriever import Retriever


@pytest.mark.parametrize("query", ["", "   \n\t  "])
def test_empty_or_whitespace_query_returns_empty_results(query):
    retriever = Retriever()

    results = retriever.retrieve(query)

    assert results == []


def test_valid_query_returns_chunks_with_text_metadata_and_score():
    retriever = Retriever()

    results = retriever.retrieve("What are the safety features of Tata Nexon?")

    assert len(results) > 0
    for result in results:
        assert "citation_id" in result
        assert "text" in result
        assert "metadata" in result
        assert "score" in result
        assert "source" in result["metadata"]
        assert "section_title" in result["metadata"]
        assert "chunk_index" in result["metadata"]


def test_retrieve_returns_chunks_sorted_by_relevance_score(monkeypatch):
    mock_results = [
        {
            "text": "Comfort features",
            "metadata": {"source": "nexon.md", "section_title": "Comfort", "chunk_index": 2},
            "score": 0.75,
        },
        {
            "text": "Safety features",
            "metadata": {"source": "nexon.md", "section_title": "Safety", "chunk_index": 0},
            "score": 0.92,
        },
        {
            "text": "Performance features",
            "metadata": {"source": "nexon.md", "section_title": "Performance", "chunk_index": 1},
            "score": 0.88,
        },
    ]

    monkeypatch.setattr(
        Retriever,
        "_search_chunks",
        lambda self, query: mock_results,
        raising=False,
    )
    retriever = Retriever()

    results = retriever.retrieve("What are the safety features of Tata Nexon?")

    assert [result["score"] for result in results] == [0.92, 0.88, 0.75]


def test_retrieve_respects_top_k_parameter(monkeypatch):
    mock_results = [
        {
            "text": "Safety overview",
            "metadata": {"source": "nexon.md", "section_title": "Safety", "chunk_index": 0},
            "score": 0.92,
        },
        {
            "text": "Airbags and braking",
            "metadata": {"source": "nexon.md", "section_title": "Safety", "chunk_index": 1},
            "score": 0.88,
        },
        {
            "text": "Structural protection",
            "metadata": {"source": "nexon.md", "section_title": "Safety", "chunk_index": 2},
            "score": 0.75,
        },
    ]

    monkeypatch.setattr(
        Retriever,
        "_search_chunks",
        lambda self, query: mock_results,
        raising=False,
    )
    retriever = Retriever()

    results = retriever.retrieve("Tata Nexon safety features", top_k=2)

    assert len(results) <= 2


def test_retrieve_uses_embedder_to_embed_query_before_search(monkeypatch):
    query = "What are the features of Tata Nexon?"
    calls = []
    embedder = Mock()
    embedder.embed.side_effect = lambda text: calls.append(("embed", text)) or [0.1, 0.2]

    def mock_search(self, query_embedding):
        calls.append(("search", query_embedding))
        return []

    monkeypatch.setattr(Retriever, "_search_chunks", mock_search)
    retriever = Retriever(embedder=embedder)

    retriever.retrieve(query)

    embedder.embed.assert_called_once_with(query)
    assert calls == [("embed", query), ("search", [0.1, 0.2])]


def test_retrieve_calls_vector_store_similarity_search_with_query_embedding():
    query_embedding = [0.1, 0.2, 0.3]
    embedder = Mock()
    embedder.embed.return_value = query_embedding
    vector_store = Mock()
    vector_store.similarity_search.return_value = []
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    retriever.retrieve("What are the features of Tata Nexon?", top_k=3)

    vector_store.similarity_search.assert_called_once_with(query_embedding, top_k=3)


def test_retrieve_end_to_end_with_mocked_embedder_and_vector_store():
    query = "What are the safety features of Tata Nexon?"
    query_embedding = [0.1, 0.2, 0.3]
    embedder = Mock()
    embedder.embed.return_value = query_embedding
    vector_store = Mock()
    vector_store.similarity_search.return_value = [
        {
            "text": "Airbags and braking systems",
            "metadata": {"source": "nexon.md", "section_title": "Safety", "chunk_index": 1},
            "score": 0.75,
        },
        {
            "text": "High-strength body structure",
            "metadata": {"source": "nexon.md", "section_title": "Safety", "chunk_index": 2},
            "score": 0.88,
        },
        {
            "text": "Advanced safety features",
            "metadata": {"source": "nexon.md", "section_title": "Safety", "chunk_index": 0},
            "score": 0.92,
        },
    ]
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    results = retriever.retrieve(query, top_k=3)

    embedder.embed.assert_called_once_with(query)
    vector_store.similarity_search.assert_called_once_with(query_embedding, top_k=3)
    assert len(results) == 3
    for result in results:
        assert "text" in result
        assert "metadata" in result
        assert "score" in result
        assert "citation_id" in result

    assert [result["score"] for result in results] == [0.92, 0.88, 0.75]


def test_retrieve_returns_empty_list_when_scores_are_below_similarity_threshold():
    query_embedding = [0.1, 0.2, 0.3]
    embedder = Mock()
    embedder.embed.return_value = query_embedding
    vector_store = Mock()
    vector_store.similarity_search.return_value = [
        {
            "text": "Unrelated warranty details",
            "metadata": {"source": "nexon.md", "section_title": "Warranty", "chunk_index": 7},
            "score": 0.31,
        },
        {
            "text": "Generic service interval details",
            "metadata": {"source": "nexon.md", "section_title": "Service", "chunk_index": 8},
            "score": 0.42,
        },
    ]
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    results = retriever.retrieve(
        "What are the safety features of Tata Nexon?",
        top_k=3,
        similarity_threshold=0.8,
    )

    assert results == []


def test_retrieve_filters_out_chunks_below_similarity_threshold():
    query_embedding = [0.1, 0.2, 0.3]
    embedder = Mock()
    embedder.embed.return_value = query_embedding
    vector_store = Mock()
    vector_store.similarity_search.return_value = [
        {
            "text": "Strong safety rating",
            "metadata": {"source": "nexon.md", "section_title": "Safety", "chunk_index": 0},
            "score": 0.92,
        },
        {
            "text": "Weakly related accessory detail",
            "metadata": {"source": "nexon.md", "section_title": "Accessories", "chunk_index": 4},
            "score": 0.79,
        },
        {
            "text": "Airbags and braking",
            "metadata": {"source": "nexon.md", "section_title": "Safety", "chunk_index": 1},
            "score": 0.8,
        },
    ]
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    results = retriever.retrieve(
        "What are the safety features of Tata Nexon?",
        similarity_threshold=0.8,
    )

    assert [result["score"] for result in results] == [0.92, 0.8]
    assert all(result["score"] >= 0.8 for result in results)


def test_retrieve_sorts_results_by_score_after_similarity_threshold_filtering():
    query_embedding = [0.1, 0.2, 0.3]
    embedder = Mock()
    embedder.embed.return_value = query_embedding
    vector_store = Mock()
    vector_store.similarity_search.return_value = [
        {
            "text": "Low relevance",
            "metadata": {"source": "nexon.md", "section_title": "General", "chunk_index": 5},
            "score": 0.72,
        },
        {
            "text": "Airbags",
            "metadata": {"source": "nexon.md", "section_title": "Safety", "chunk_index": 1},
            "score": 0.86,
        },
        {
            "text": "Braking systems",
            "metadata": {"source": "nexon.md", "section_title": "Safety", "chunk_index": 2},
            "score": 0.81,
        },
        {
            "text": "Safety overview",
            "metadata": {"source": "nexon.md", "section_title": "Safety", "chunk_index": 0},
            "score": 0.95,
        },
    ]
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    results = retriever.retrieve(
        "What are the safety features of Tata Nexon?",
        similarity_threshold=0.8,
    )

    assert [result["score"] for result in results] == [0.95, 0.86, 0.81]


def test_retrieve_returns_chunks_with_citation_ids_and_page_metadata():
    retriever = Retriever()

    results = retriever.retrieve("What are the safety features of Tata Nexon?")

    assert len(results) > 0
    for result in results:
        metadata = result["metadata"]
        assert result["citation_id"] == f"{metadata['source']}:{metadata['chunk_index']}"
        assert "page_number" in metadata
