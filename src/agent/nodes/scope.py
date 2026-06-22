"""Shared scope helpers for Tata Nexon-only agent nodes.

This module keeps product-scope decisions consistent across the input
guardrail and router. The input guardrail remains the authoritative gatekeeper,
while the router can use the same helpers as a defensive fallback when called
directly in tests or tooling.
"""

from __future__ import annotations

import re
from typing import Optional


COMPARISON_RESPONSE_TEMPLATE = (
    "I'm a specialized assistant for the Tata Nexon only. I don't have "
    "information or comparison data for other models like {target}. Would you "
    "like to know more about any specific feature of the Tata Nexon?"
)

NEXON_REFERENCES = (
    "tata nexon",
    "nexon",
    "this car",
    "the car",
    "this vehicle",
    "the vehicle",
    "this suv",
    "the suv",
    "this model",
    "the model",
)

TRAILING_COMPARISON_TOPICS = (
    "performance",
    "price",
    "features",
    "feature",
    "safety",
    "mileage",
    "warranty",
    "coverage",
    "service",
    "spec",
    "specs",
    "specification",
    "specifications",
    "details",
    "comparison",
)

NEXON_ONLY_COMPARISON_TARGETS = (
    "variant",
    "variants",
    "trim",
    "trims",
    "model",
    "models",
)


def comparison_response(target: Optional[str]) -> str:
    """Return the standard polite refusal for external comparison queries.

    Args:
        target: External car/model name extracted from the user query.

    Returns:
        User-facing message that explains the Tata Nexon-only scope.
    """
    return COMPARISON_RESPONSE_TEMPLATE.format(target=target or "other car models")


def comparison_target(query: str) -> Optional[str]:
    """Extract an external comparison target from a Tata Nexon query.

    Args:
        query: Raw user query.

    Returns:
        Display-ready external model name when the query asks to compare Tata
        Nexon with another car, otherwise ``None``.
    """
    if not query:
        return None

    for pattern in _comparison_patterns():
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue

        target = _clean_comparison_target(match.group("target"))
        if target:
            return target
    return None


def target_from_blocked_reason(blocked_reason: Optional[str]) -> str:
    """Extract a display target from a comparison blocking reason.

    Args:
        blocked_reason: Guardrail blocking reason text.

    Returns:
        Extracted model name or a generic fallback.
    """
    if blocked_reason:
        match = re.search(r"Comparison with (?P<target>.+?) is outside", blocked_reason)
        if match:
            return match.group("target")
    return "other car models"


def _comparison_patterns() -> tuple[str, ...]:
    """Return regex patterns for common comparison phrasings.

    Returns:
        Ordered regular expressions that capture an external comparison target.
    """
    nexon = _nexon_reference_pattern()
    target = r"(?P<target>[A-Za-z0-9][A-Za-z0-9 .&/-]{1,80})"
    return (
        rf"\bcompare\b.*?\b{nexon}\b\s+(?:with|against|to)\s+{target}",
        rf"\bcompare\b\s+{target}\s+(?:with|against|to)\s+\b{nexon}\b",
        rf"\b{nexon}\b\s+(?:vs\.?|versus|against)\s+{target}",
        rf"\b{target}\s+(?:vs\.?|versus|against)\s+{nexon}\b",
        rf"\b(?:is|are)?\s*{nexon}\b\s+better than\s+{target}",
        rf"\b(?:is|are)?\s*{target}\s+better than\s+{nexon}\b",
        rf"\bdifference between\s+{nexon}\b\s+(?:and|&)\s+{target}",
        rf"\bdifference between\s+{target}\s+(?:and|&)\s+{nexon}\b",
        rf"\b{nexon}\b\s+compared to\s+{target}",
        rf"\b{target}\s+compared to\s+{nexon}\b",
    )


def _nexon_reference_pattern() -> str:
    """Build a regex alternation for explicit and implicit Nexon references.

    Returns:
        Regex string matching allowed references to Tata Nexon.
    """
    references = "|".join(re.escape(reference) for reference in NEXON_REFERENCES)
    return f"(?:{references})"


def _clean_comparison_target(raw_target: str) -> Optional[str]:
    """Normalize a captured comparison target and reject Nexon-only targets.

    Args:
        raw_target: Raw target text captured by a comparison regex.

    Returns:
        Display-ready target name, or ``None`` when the target is still within
        Tata Nexon scope such as variant or trim comparisons.
    """
    target = re.sub(r"[?!.:,;]+$", "", raw_target or "").strip()
    target = re.sub(r"\s+", " ", target)
    if not target:
        return None

    words = target.split()
    while words and words[-1].lower() in TRAILING_COMPARISON_TOPICS:
        words.pop()
    target = " ".join(words).strip()
    if not target:
        return None

    lower_target = target.lower()
    if any(reference == lower_target for reference in NEXON_REFERENCES):
        return None
    if lower_target in NEXON_ONLY_COMPARISON_TARGETS:
        return None
    return _display_model_name(target)


def _display_model_name(target: str) -> str:
    """Format a model name while preserving common acronyms.

    Args:
        target: Normalized target text.

    Returns:
        Human-readable model name for refusal text.
    """
    display_words = []
    for word in target.split():
        clean_word = word.strip()
        lower_word = clean_word.lower()
        if lower_word in {"ev", "cng", "xuv", "suv"} or any(char.isdigit() for char in clean_word):
            display_words.append(clean_word.upper())
        elif clean_word.isupper():
            display_words.append(clean_word)
        else:
            display_words.append(clean_word.capitalize())
    return " ".join(display_words)
