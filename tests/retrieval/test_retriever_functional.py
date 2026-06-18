import os
from pathlib import Path
from uuid import uuid4

import pytest
from dotenv import load_dotenv

from src.ingestion.embedder import Embedder
from src.ingestion.storer import VectorStorer
from src.retrieval.retriever import Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.txt")


SAMPLE_CHUNKS = [
    {
        "text": (
            "Tata Nexon safety features include six airbags, electronic stability "
            "program, strong body structure, hill hold control, and braking support."
        ),
        "metadata": {
            "source": "functional-tata-nexon.md",
            "section_title": "Safety",
            "chunk_index": 0,
            "page_number": 2,
        },
    },
    {
        "text": (
            "Tata Nexon performance highlights include a responsive turbo petrol "
            "engine, diesel engine option, drive modes, and efficient mileage."
        ),
        "metadata": {
            "source": "functional-tata-nexon.md",
            "section_title": "Performance",
            "chunk_index": 1,
            "page_number": 3,
        },
    },
    {
        "text": (
            "Tata Nexon comfort features include a spacious cabin, connected car "
            "technology, infotainment display, ventilated seats, and premium audio."
        ),
        "metadata": {
            "source": "functional-tata-nexon.md",
            "section_title": "Comfort",
            "chunk_index": 2,
            "page_number": 4,
        },
    },
]


@pytest.fixture(scope="module")
def openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip(
            "OPENAI_API_KEY was not found in the environment, .env, or .env.txt; "
            "skipping real OpenAI retrieval functional tests."
        )
    return api_key


@pytest.fixture(scope="module")
def real_embedder(openai_api_key: str) -> Embedder:
    embedder = Embedder.from_env()
    assert embedder.api_key == openai_api_key
    return embedder


@pytest.fixture(scope="module")
def real_vector_storer(tmp_path_factory, real_embedder: Embedder) -> VectorStorer:
    storer = VectorStorer(
        collection_name=f"retrieval-functional-{uuid4().hex[:8]}",
        persist_directory=str(tmp_path_factory.mktemp("chroma-retrieval-functional")),
    )
    chunks = [
        {
            "text": chunk["text"],
            "metadata": dict(chunk["metadata"]),
        }
        for chunk in SAMPLE_CHUNKS
    ]
    embeddings = real_embedder.embed_batch([chunk["text"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        assert embedding, "OpenAI Embedder returned an empty embedding for sample data."
        chunk["embedding"] = embedding

    storer.store(chunks)
    return storer


@pytest.fixture(scope="module")
def real_retriever(
    real_embedder: Embedder,
    real_vector_storer: VectorStorer,
) -> Retriever:
    return Retriever(embedder=real_embedder, vector_store=real_vector_storer)


def test_retriever_functional_end_to_end_with_real_openai_embedder(
    real_retriever: Retriever,
):
    results = real_retriever.retrieve(
        "What are the safety features of Tata Nexon?",
        top_k=3,
    )

    assert results
    for result in results:
        assert result["text"]
        assert "metadata" in result
        assert isinstance(result["score"], float)
        assert result["citation_id"] == (
            f"{result['metadata']['source']}:{result['metadata']['chunk_index']}"
        )


def test_retriever_functional_top_k_limits_real_results(
    real_retriever: Retriever,
):
    results = real_retriever.retrieve("Tata Nexon safety features", top_k=2)

    assert len(results) <= 2


def test_retriever_functional_similarity_threshold_filters_real_results(
    real_retriever: Retriever,
):
    baseline_results = real_retriever.retrieve("Tata Nexon safety features", top_k=3)
    assert len(baseline_results) == 3

    scores = [result["score"] for result in baseline_results]
    if scores[0] == scores[-1]:
        pytest.skip("Real embeddings returned tied scores; threshold filtering cannot be asserted.")

    threshold = (scores[0] + scores[1]) / 2
    filtered_results = real_retriever.retrieve(
        "Tata Nexon safety features",
        top_k=3,
        similarity_threshold=threshold,
    )

    assert filtered_results
    assert len(filtered_results) < len(baseline_results)
    assert all(result["score"] >= threshold for result in filtered_results)
    assert [result["score"] for result in filtered_results] == sorted(
        [result["score"] for result in filtered_results],
        reverse=True,
    )
