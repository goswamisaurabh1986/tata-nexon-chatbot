def chunk(index: int, score: float, text: str = "Safety features") -> dict:
    return {
        "citation_id": f"Tata_Nexon_Brochure.pdf:{index}",
        "text": text,
        "metadata": {
            "source": "Tata_Nexon_Brochure.pdf",
            "section_title": "Safety",
            "chunk_index": index,
        },
        "score": score,
    }


def test_grade_node_returns_graded_chunks_with_relevance_scores():
    from src.agent.nodes.grade_node import GradeNode

    node = GradeNode(grader_llm=None)
    state = {"retrieved_chunks": [chunk(0, 0.91), chunk(1, 0.82)]}

    result = node.run(state)

    assert len(result["graded_chunks"]) == 2
    for graded_chunk in result["graded_chunks"]:
        assert "relevance_score" in graded_chunk
        assert 0.0 <= graded_chunk["relevance_score"] <= 1.0


def test_grade_node_routes_high_quality_chunks_to_generate():
    from src.agent.nodes.grade_node import GradeNode

    node = GradeNode(grader_llm=None, min_relevance_score=0.7, min_relevant_chunks=2)
    state = {"retrieved_chunks": [chunk(0, 0.94), chunk(1, 0.81)]}

    result = node.run(state)

    assert result["route"] == "generate"


def test_grade_node_routes_poor_or_insufficient_chunks_to_clarify_or_rewrite():
    from src.agent.nodes.grade_node import GradeNode

    node = GradeNode(grader_llm=None, min_relevance_score=0.7, min_relevant_chunks=2)
    state = {"retrieved_chunks": [chunk(0, 0.45), chunk(1, 0.22)]}

    result = node.run(state)

    assert result["route"] in {"clarify", "rewrite"}


def test_grade_node_adds_reasoning_step_for_grading_decision():
    from src.agent.nodes.grade_node import GradeNode

    node = GradeNode(grader_llm=None, min_relevance_score=0.7, min_relevant_chunks=1)
    state = {
        "retrieved_chunks": [chunk(0, 0.88)],
        "reasoning_steps": ["Retrieved 1 relevant chunks."],
    }

    result = node.run(state)

    assert result["reasoning_steps"] == [
        "Retrieved 1 relevant chunks.",
        "Graded retrieval context: 1 chunks passed relevance threshold.",
    ]


def test_grade_node_filters_out_very_low_relevance_chunks():
    from src.agent.nodes.grade_node import GradeNode

    node = GradeNode(grader_llm=None, min_relevance_score=0.7, filter_threshold=0.3)
    state = {
        "retrieved_chunks": [
            chunk(0, 0.91),
            chunk(1, 0.29, text="Unrelated service detail"),
            chunk(2, 0.74),
        ]
    }

    result = node.run(state)

    assert [graded["citation_id"] for graded in result["graded_chunks"]] == [
        "Tata_Nexon_Brochure.pdf:0",
        "Tata_Nexon_Brochure.pdf:2",
    ]


def test_grade_node_handles_no_retrieved_chunks():
    from src.agent.nodes.grade_node import GradeNode

    node = GradeNode(grader_llm=None)
    state = {"retrieved_chunks": []}

    result = node.run(state)

    assert result["graded_chunks"] == []
    assert result["route"] == "clarify"
    assert result["reasoning_steps"] == [
        "No retrieved chunks available for grading.",
    ]
