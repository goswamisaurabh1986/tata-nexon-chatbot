from src.ingestion.parsers.text_parser import TextParser


def test_parser_extracts_sections_from_tata_nexon_text(simple_section_document):
    parsed_document = TextParser().parse(simple_section_document)

    assert parsed_document["sections"] == [
        {
            "title": "Safety",
            "content": "Advanced safety features...",
            "level": None,
        },
        {
            "title": "Performance",
            "content": "Powerful engine...",
            "level": None,
        },
    ]
    assert parsed_document["metadata"]["section_count"] == 2
