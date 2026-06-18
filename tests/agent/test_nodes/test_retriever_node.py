from unittest.mock import Mock


def sample_chunk(index: int, score: float = 0.9) -> dict:
    return {
        "citation_id": f"Tata_Nexon_Brochure.pdf:{index}",
        "text": f"Sample retrieved chunk {index}",
        "metadata": {
            "source": "Tata_Nexon_Brochure.pdf",
            "section_title": "Safety",
            "chunk_index": index,
        },
        "score": score,
    }


def test_retriever_node_calls_retrieval_module_and_returns_retrieved_chunks():
    from src.agent.nodes.retriever_node import RetrieverNode

    retriever = Mock()
    chunks = [sample_chunk(0), sample_chunk(1)]
    retriever.retrieve.return_value = chunks
    node = RetrieverNode(retriever=retriever, top_k=2)

    result = node.run({"query": "What are the safety features of Tata Nexon?"})

    retriever.retrieve.assert_called_once_with(
        "What are the safety features of Tata Nexon?",
        top_k=2,
    )
    assert result["retrieved_chunks"] == chunks


def test_retriever_node_stores_citation_ids_from_retriever_output():
    from src.agent.nodes.retriever_node import RetrieverNode

    retriever = Mock()
    retriever.retrieve.return_value = [sample_chunk(0), sample_chunk(3)]
    node = RetrieverNode(retriever=retriever)

    result = node.run({"query": "Tata Nexon safety features"})

    assert result["citations"] == [
        {"citation_id": "Tata_Nexon_Brochure.pdf:0"},
        {"citation_id": "Tata_Nexon_Brochure.pdf:3"},
    ]


def test_retriever_node_handles_empty_retrieval_results_gracefully():
    from src.agent.nodes.retriever_node import RetrieverNode

    retriever = Mock()
    retriever.retrieve.return_value = []
    node = RetrieverNode(retriever=retriever)

    result = node.run({"query": "Tata Nexon safety features"})

    assert result["retrieved_chunks"] == []
    assert result["citations"] == []
    assert result["route"] == "clarify"
    assert result["error"] is None


def test_retriever_node_respects_top_k_parameter():
    from src.agent.nodes.retriever_node import RetrieverNode

    retriever = Mock()
    retriever.retrieve.return_value = [sample_chunk(0)]
    node = RetrieverNode(retriever=retriever, top_k=1)

    node.run({"query": "Tata Nexon safety features"})

    retriever.retrieve.assert_called_once_with("Tata Nexon safety features", top_k=1)


def test_retriever_node_updates_route_and_adds_reasoning_step():
    from src.agent.nodes.retriever_node import RetrieverNode

    retriever = Mock()
    retriever.retrieve.return_value = [sample_chunk(0)]
    node = RetrieverNode(retriever=retriever)

    result = node.run(
        {
            "query": "Tata Nexon safety features",
            "reasoning_steps": ["Query requires retrieval."],
        }
    )

    assert result["route"] == "retrieval"
    assert result["reasoning_steps"] == [
        "Query requires retrieval.",
        "Retrieved 1 relevant chunks.",
    ]
