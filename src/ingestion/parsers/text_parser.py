import re
from typing import Optional


class TextParser:
    """Clean plain text and extract document sections for downstream chunking."""

    HEADING_PATTERNS = (
        r"^#{1,6}\s+(?P<title>.+)$",
        r"^(?:\d+(?:\.\d+)*[.)]?)\s+(?P<title>.+)$",
        r"^(?P<title>[A-Z][A-Za-z0-9 /&(),-]{1,100}):$",
        r"^(?P<title>[A-Z][A-Za-z0-9 /&(),-]*\s+Section)$",
    )
    ENCODING_FIXES = {
        "\u00e2\u20ac\u201d": "-",
        "\u00e2\u20ac\u201c": "-",
        "\u00e2\u20ac\u2122": "'",
        "\u00e2\u20ac\u0153": '"',
        "\u00e2\u20ac\u009d": '"',
        "\u00c2": "",
    }

    def parse(self, raw_text: str) -> dict:
        """Return cleaned text, extracted sections, and lightweight metadata."""
        cleaned_text, metadata = self._clean_text(raw_text)
        sections = self._extract_sections(cleaned_text)

        metadata.update(
            {
                "section_count": len(sections),
                "heading_count": sum(1 for section in sections if section["title"]),
                "word_count": len(cleaned_text.split()),
            }
        )

        return {
            "cleaned_text": cleaned_text,
            "sections": sections,
            "metadata": metadata,
        }

    def _clean_text(self, text: str) -> tuple[str, dict]:
        text = self._fix_encoding(text)
        text = self._fix_pdf_line_breaks(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        raw_lines = text.split("\n")
        repeated_lines = self._repeated_header_footer_lines(raw_lines)
        cleaned_lines = []
        removed_page_numbers = 0
        removed_repeated_lines = 0
        for line in raw_lines:
            cleaned_line = self._clean_line(line)
            if self._is_page_number(cleaned_line):
                removed_page_numbers += 1
                continue
            if cleaned_line in repeated_lines:
                removed_repeated_lines += 1
                continue
            cleaned_lines.append(cleaned_line)

        cleaned_text = "\n".join(cleaned_lines).strip()
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        return cleaned_text, {
            "page_numbers_removed": removed_page_numbers,
            "repeated_header_footer_lines_removed": removed_repeated_lines,
        }

    def _fix_encoding(self, text: str) -> str:
        for broken, fixed in self.ENCODING_FIXES.items():
            text = text.replace(broken, fixed)
        return text

    def _fix_pdf_line_breaks(self, text: str) -> str:
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        return re.sub(r"(?<=[a-z])(?=[A-Z][a-z])", " ", text)

    def _clean_line(self, line: str) -> str:
        line = re.sub(r"[ \t]+", " ", line).strip()
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        return line

    def _repeated_header_footer_lines(self, lines: list[str]) -> set[str]:
        normalized_lines = [self._clean_line(line) for line in lines]
        counts = {}
        for line in normalized_lines:
            if line and len(line) <= 80:
                counts[line] = counts.get(line, 0) + 1
        return {line for line, count in counts.items() if count > 1 and not self._is_heading(line)}

    def _is_page_number(self, line: str) -> bool:
        return any(
            re.match(pattern, line, flags=re.IGNORECASE)
            for pattern in (
                r"^page\s+\d+(?:\s+of\s+\d+)?$",
                r"^-\s*\d+\s*-$",
                r"^\d+$",
            )
        )

    def _extract_sections(self, text: str) -> list[dict]:
        sections = []
        pending_title = None
        pending_level = None
        pending_content = []

        for block in self._paragraph_blocks(text):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue

            if len(lines) == 1 and self._is_heading(lines[0]):
                self._flush_pending_section(sections, pending_title, pending_content, pending_level)
                pending_title, pending_level = self._heading_info(lines[0])
                pending_content = []
                continue

            title, level = self._heading_info(lines[0])
            if title and len(lines) > 1:
                self._flush_pending_section(sections, pending_title, pending_content, pending_level)
                sections.append(self._section(title, lines[1:], level))
                pending_title = None
                pending_level = None
                pending_content = []
            elif pending_title:
                pending_content.extend(lines)

        self._flush_pending_section(sections, pending_title, pending_content, pending_level)
        return sections

    def _paragraph_blocks(self, text: str) -> list[str]:
        return [block.strip() for block in text.split("\n\n") if block.strip()]

    def _flush_pending_section(
        self,
        sections: list[dict],
        title: Optional[str],
        content_lines: list[str],
        level: Optional[int],
    ) -> None:
        if title and content_lines:
            sections.append(self._section(title, content_lines, level))

    def _section(self, title: str, content_lines: list[str], level: Optional[int]) -> dict:
        return {
            "title": title,
            "content": "\n".join(content_lines).strip(),
            "level": level,
        }

    def _heading_info(self, line: str) -> tuple[Optional[str], Optional[int]]:
        if not self._is_heading(line):
            return None, None

        for pattern in self.HEADING_PATTERNS:
            match = re.match(pattern, line)
            if match:
                return match.group("title").strip(), self._heading_level(line)

        return line.rstrip(":").strip(), None

    def _is_heading(self, line: str) -> bool:
        if not line or len(line) > 120 or line.endswith((".", "!", "?")):
            return False

        if any(re.match(pattern, line) for pattern in self.HEADING_PATTERNS):
            return True

        words = line.rstrip(":").split()
        return 1 <= len(words) <= 6 and line[:1].isupper()

    def _heading_level(self, line: str) -> Optional[int]:
        markdown_match = re.match(r"^(#{1,6})\s+", line)
        if markdown_match:
            return len(markdown_match.group(1))

        numbered_match = re.match(r"^(\d+(?:\.\d+)*)", line)
        if numbered_match:
            return numbered_match.group(1).count(".") + 1

        return None
