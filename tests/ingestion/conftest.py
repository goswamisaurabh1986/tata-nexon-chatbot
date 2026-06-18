from unittest.mock import Mock

import fitz
import pytest


@pytest.fixture
def simple_section_document():
    return (
        "Tata Nexon Features\n\n"
        "Safety\n"
        "Advanced safety features...\n\n"
        "Performance\n"
        "Powerful engine..."
    )


@pytest.fixture
def section_document():
    return (
        "Tata Nexon Features\n\n"
        "Safety Section\n"
        "Advanced safety features including 6 airbags.\n\n"
        "Performance Section\n"
        "Powerful 1.2L turbo engine with great mileage."
    )


@pytest.fixture
def long_document():
    return (
        "The Tata Nexon combines a bold design with practical cabin space, "
        "connected technology, confident ride quality, and strong safety "
        "equipment for daily driving. Its feature list includes modern "
        "infotainment, comfortable seating, useful storage, and convenience "
        "features for city commutes and highway trips. The vehicle balances "
        "performance, mileage, and durability while keeping passenger comfort "
        "and safety at the center of the ownership experience. "
    ) * 2


@pytest.fixture
def mock_embedder():
    embedder = Mock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    return embedder


@pytest.fixture
def mock_batch_embedder():
    embedder = Mock()
    embedder.embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]
    return embedder


@pytest.fixture
def mock_storer():
    return Mock()


@pytest.fixture
def embedded_chunks():
    return [
        {
            "text": "Advanced safety features including 6 airbags.",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {
                "source": "nexon-brochure.pdf",
                "section_title": "Safety Section",
                "chunk_index": 0,
                "total_chunks": 2,
                "chunk_size": 200,
            },
        },
        {
            "text": "Powerful 1.2L turbo engine with great mileage.",
            "embedding": [0.4, 0.5, 0.6],
            "metadata": {
                "source": "nexon-brochure.pdf",
                "section_title": "Performance Section",
                "chunk_index": 1,
                "total_chunks": 2,
                "chunk_size": 200,
            },
        },
    ]


@pytest.fixture
def write_pdf():
    def _write_pdf(path, pages):
        document = fitz.open()
        for page_text in pages:
            page = document.new_page()
            page.insert_text((72, 72), page_text)
        document.save(path)
        document.close()

    return _write_pdf
