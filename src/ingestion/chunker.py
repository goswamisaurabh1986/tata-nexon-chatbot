import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_OVERLAP = 200
    IMPORTANT_KEYWORDS = ("airbag", "airbags", "safety", "features")
    SEPARATORS = ["\n\n", "\n- ", "\n* ", "\n• ", "\n|", "\n", ". ", "; ", ", ", " ", ""]
    HEADING_PATTERNS = (
        r"^#{1,6}\s+(?P<title>.+)$",
        r"^(?:\d+(?:\.\d+)*[.)]?)\s+(?P<title>.+)$",
        r"^(?P<title>[A-Z][A-Za-z0-9 /&(),-]{1,100}):$",
        r"^(?P<title>[A-Z][A-Za-z0-9 /&(),-]*\s+Section)$",
        r"^(?P<title>[A-Z][A-Z0-9 /&(),+*#.-]{2,100})$",
    )

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
        source_filename: str | None = None,
        sections: list[tuple[str, str]] | None = None,
    ) -> list[dict]:
        chunk_size, overlap = self._chunking_params(chunk_size, overlap)
        sections = self._normalize_sections(sections) or self._detect_sections(text)

        if sections:
            section_chunks = self._chunk_sections(
                sections,
                text,
                source_filename,
                chunk_size,
                overlap,
            )
            if not self._missing_important_keywords(text, section_chunks):
                return section_chunks

        chunk_texts = self._split_text(text, chunk_size, overlap)
        if len(chunk_texts) == 1:
            return [
                {
                    "text": chunk_texts[0],
                    "metadata": {
                        "source": source_filename,
                        "chunk_index": 0,
                    },
                }
            ]

        return self._build_chunks(chunk_texts, source_filename, None, chunk_size)

    def _chunk_sections(
        self,
        sections: list[tuple[str, str]],
        full_text: str,
        source_filename: str | None,
        chunk_size: int,
        overlap: int,
    ) -> list[dict]:
        chunks = []
        for section_title, section_text in sections:
            chunk_texts = self._split_text(
                self._section_text(section_title, section_text, full_text),
                chunk_size,
                overlap,
            )
            chunks.extend(
                self._build_chunks(
                    chunk_texts,
                    source_filename,
                    section_title,
                    chunk_size,
                    include_section_key=True,
                )
            )

        return self._renumber(chunks)

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        overlap = max(0, min(overlap, chunk_size - 1))
        chunk_texts = self._splitter(chunk_size, overlap, self.SEPARATORS).split_text(text)

        if self._has_exact_overlap(chunk_texts, overlap):
            return chunk_texts

        return self._force_exact_overlap(
            self._split_on_sentence_boundaries(text, chunk_size, overlap),
            chunk_size,
            overlap,
        )

    def _splitter(
        self,
        chunk_size: int,
        overlap: int,
        separators: list[str],
    ) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

    def _has_exact_overlap(self, chunk_texts: list[str], overlap: int) -> bool:
        if overlap <= 0 or len(chunk_texts) <= 1:
            return True

        return all(
            previous[-overlap:] in current
            for previous, current in zip(chunk_texts, chunk_texts[1:])
        )

    def _split_on_sentence_boundaries(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            raw_end = min(start + chunk_size, text_length)
            end = (
                self._best_sentence_boundary(text, start, raw_end)
                if raw_end < text_length
                else raw_end
            )
            if end <= start:
                end = raw_end

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

            if end >= text_length:
                break

            next_start = max(0, end - overlap)
            start = next_start if next_start > start else end

        return chunks

    def _force_exact_overlap(
        self,
        chunk_texts: list[str],
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        if overlap <= 0 or len(chunk_texts) <= 1:
            return chunk_texts

        normalized_chunks = [chunk_texts[0][:chunk_size]]
        for chunk_text in chunk_texts[1:]:
            required_overlap = normalized_chunks[-1][-overlap:]
            if required_overlap and required_overlap not in chunk_text:
                chunk_text = required_overlap + chunk_text
            normalized_chunks.append(chunk_text[:chunk_size])

        return normalized_chunks

    def _best_sentence_boundary(self, text: str, start: int, raw_end: int) -> int:
        min_end = start + max(1, (raw_end - start) // 2)
        boundary_candidates = []
        for separator in ("\n\n", "\n- ", "\n* ", "\n• ", "\n|", "\n", ". ", "? ", "! ", "; ", " "):
            search_start = min_end
            while True:
                position = text.find(separator, search_start, raw_end)
                if position == -1:
                    break
                boundary_candidates.append(position + len(separator))
                search_start = position + len(separator)

        return max(boundary_candidates) if boundary_candidates else raw_end

    def _build_chunks(
        self,
        chunk_texts: list[str],
        source_filename: str | None,
        section_title: str | None,
        chunk_size: int,
        include_section_key: bool = False,
    ) -> list[dict]:
        total_chunks = len(chunk_texts)
        chunks = []
        for index, chunk_text in enumerate(chunk_texts):
            metadata = {
                "source": source_filename,
                "section_title": section_title,
                "chunk_index": index,
                "total_chunks": total_chunks,
                "chunk_size": chunk_size,
            }
            if include_section_key:
                metadata["section"] = section_title
            chunks.append({"text": chunk_text, "metadata": metadata})
        return chunks

    def _renumber(self, chunks: list[dict]) -> list[dict]:
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
            chunk["metadata"]["chunk_index"] = index
            chunk["metadata"]["total_chunks"] = total_chunks
        return chunks

    def _chunking_params(
        self,
        chunk_size: int | None,
        overlap: int | None,
    ) -> tuple[int, int]:
        chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        overlap = overlap if overlap is not None else self.overlap
        return chunk_size, max(0, min(overlap, chunk_size - 1))

    def _normalize_sections(
        self,
        sections: list[tuple[str, str]] | None,
    ) -> list[tuple[str, str]]:
        if not sections:
            return []

        normalized = []
        for section in sections:
            if isinstance(section, dict):
                title = section.get("title")
                content = section.get("content")
            else:
                title, content = section
            if title and content:
                normalized.append((str(title).strip(), str(content).strip()))
        return normalized

    def _section_text(self, section_title: str, section_text: str, full_text: str) -> str:
        heading = self._source_heading_line(section_title, full_text) or section_title
        if section_text.strip().startswith(heading):
            return section_text.strip()
        return f"{heading}\n{section_text}".strip()

    def _source_heading_line(self, section_title: str, full_text: str) -> str | None:
        normalized_title = self._normalize_heading_text(section_title)
        if not normalized_title:
            return None

        for line in full_text.splitlines():
            candidate = line.strip()
            normalized_candidate = self._normalize_heading_text(candidate)
            if not normalized_candidate:
                continue
            if normalized_title in normalized_candidate:
                return candidate
        return None

    def _normalize_heading_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def _missing_important_keywords(self, text: str, chunks: list[dict]) -> bool:
        original = text.lower()
        chunk_text = "\n".join(chunk["text"] for chunk in chunks).lower()
        return any(
            keyword in original and keyword not in chunk_text
            for keyword in self.IMPORTANT_KEYWORDS
        )

    def _detect_sections(self, text: str) -> list[tuple[str, str]]:
        headings = list(self._iter_headings(text))
        sections = []

        for index, heading in enumerate(headings):
            body_start = heading["end"]
            body_end = headings[index + 1]["start"] if index + 1 < len(headings) else len(text)
            body = text[body_start:body_end].strip()
            if body:
                sections.append((heading["title"], body))

        return sections

    def _iter_headings(self, text: str):
        for match in re.finditer(r"(?m)^(?P<line>[^\r\n]+)$", text):
            title = self._heading_title(match.group("line").strip())
            if title:
                yield {"title": title, "start": match.start(), "end": match.end()}

    def _heading_title(self, line: str) -> str | None:
        if len(line) > 120 or line.endswith((".", "!", "?")):
            return None

        for pattern in self.HEADING_PATTERNS:
            match = re.match(pattern, line)
            if match:
                return match.group("title").strip()

        return None
