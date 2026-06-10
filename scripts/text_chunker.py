"""
Hlokk - Text Chunker
Semantic chunking for academic papers, respecting section boundaries.
"""
import re
from typing import Optional


class Chunk:
    __slots__ = ("content", "metadata")

    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata  # source, page, section, chunk_id

    def to_dict(self):
        return {"content": self.content, "metadata": self.metadata}


def estimate_tokens(text: str) -> int:
    """
    Conservative token estimation. Tuned to slightly over-estimate so the
    chunker stays under embedding API hard limits (e.g. 8192) for content
    with many short tokens (IDs, numbers, references in scientific papers).

    Heuristics:
      - English words ≈ 1.6 tokens (was 1.3 — underestimated for technical text)
      - CJK chars ≈ 1.0 token each (was 0.7 — modern BPE often uses 1 token/char)
      - Always at least char/4, which is a widely cited safe upper bound for BPE.
    """
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))
    words = len(text.split())
    word_estimate = int(words * 1.6 + cjk_chars * 1.0)
    char_estimate = len(text) // 4
    return max(word_estimate, char_estimate)


def _split_by_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


def chunk_text(
    text: str,
    source_file: str,
    section: str = "unknown",
    start_page: int = 1,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """
    Chunk text into segments respecting paragraph boundaries.
    Uses a sliding-window approach with paragraph-level granularity.
    """
    paragraphs = _split_by_paragraphs(text)
    chunks = []
    current_parts = []
    current_tokens = 0
    chunk_idx = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        # if single paragraph exceeds max, split by sentences
        if para_tokens > max_tokens:
            # flush current
            if current_parts:
                chunks.append(_make_chunk(
                    current_parts, source_file, section, start_page, chunk_idx
                ))
                chunk_idx += 1
                current_parts, current_tokens = _apply_overlap(
                    current_parts, overlap_tokens
                )

            sentence_chunks = _chunk_long_paragraph(
                para, source_file, section, start_page, chunk_idx, max_tokens
            )
            chunks.extend(sentence_chunks)
            chunk_idx += len(sentence_chunks)
            current_parts = []
            current_tokens = 0
            continue

        if current_tokens + para_tokens > max_tokens and current_parts:
            chunks.append(_make_chunk(
                current_parts, source_file, section, start_page, chunk_idx
            ))
            chunk_idx += 1
            current_parts, current_tokens = _apply_overlap(
                current_parts, overlap_tokens
            )

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        chunks.append(_make_chunk(
            current_parts, source_file, section, start_page, chunk_idx
        ))

    return chunks


def _make_chunk(
    parts: list[str], source: str, section: str, page: int, idx: int
) -> Chunk:
    content = "\n\n".join(parts)
    metadata = {
        "source_file": source,
        "section": section,
        "estimated_page": page,
        "chunk_id": f"{source}::{section}::{idx}",
    }
    return Chunk(content=content, metadata=metadata)


def _apply_overlap(
    parts: list[str], overlap_tokens: int
) -> tuple[list[str], int]:
    """Keep tail paragraphs that fit within overlap budget."""
    if overlap_tokens <= 0:
        return [], 0
    kept = []
    tokens = 0
    for p in reversed(parts):
        t = estimate_tokens(p)
        if tokens + t > overlap_tokens:
            break
        kept.insert(0, p)
        tokens += t
    return kept, tokens


def _chunk_long_paragraph(
    para: str, source: str, section: str, page: int, start_idx: int,
    max_tokens: int,
) -> list[Chunk]:
    """Split an oversized paragraph by sentences, then by characters as a hard fallback."""
    sentences = re.split(r"(?<=[.!?。！？])\s+", para)
    chunks = []
    current = []
    current_tokens = 0
    idx = start_idx

    def _flush():
        nonlocal current, current_tokens, idx
        if current:
            chunks.append(_make_chunk(current, source, section, page, idx))
            idx += 1
            current = []
            current_tokens = 0

    for sent in sentences:
        # Hard fallback: a single "sentence" still exceeds max_tokens
        # (typical for table dumps or references with no sentence delimiters).
        # Force-split by character window approximating the token budget.
        if estimate_tokens(sent) > max_tokens:
            _flush()
            # ~4 chars per token is a safe BPE upper bound; include 10% headroom.
            char_window = max(200, int(max_tokens * 4 * 0.9))
            for i in range(0, len(sent), char_window):
                segment = sent[i : i + char_window]
                if segment.strip():
                    chunks.append(_make_chunk([segment], source, section, page, idx))
                    idx += 1
            continue

        st = estimate_tokens(sent)
        if current_tokens + st > max_tokens and current:
            _flush()
        current.append(sent)
        current_tokens += st

    _flush()
    return chunks


def chunk_document_sections(
    sections: dict[str, str],
    source_file: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Chunk a document's sections, preserving section metadata."""
    all_chunks = []
    for section_name, content in sections.items():
        section_chunks = chunk_text(
            text=content,
            source_file=source_file,
            section=section_name,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        all_chunks.extend(section_chunks)
    return all_chunks
