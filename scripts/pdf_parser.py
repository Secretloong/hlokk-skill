"""
Hlokk - PDF Parser & Document Classifier
Extracts text from PDFs and classifies them as main paper vs supplements.
"""
import json
import re
from pathlib import Path
from typing import Optional

import pdfplumber


# ---------- classification patterns ----------
_SUPP_PATTERNS = re.compile(
    r"(supplement|supplementary|supp[\._\-]|extended[\._\-]data|"
    r"appendix|additional[\._\-]file|supporting[\._\-]info|"
    r"table\s*s\d|figure\s*s\d|SI[\._\-]|"
    # Common journal suffix conventions:
    # Science: *_sm.pdf ; AAAS variants: *_si.pdf
    # Cell Press: *-mmc\d+.pdf ; Nature/Elsevier: *-supp*.pdf
    r"[_\-]sm\b|[_\-]si\b|[_\-]mmc\d|[_\-]supp\b)",
    re.IGNORECASE,
)

# First-page content heuristic — used as a fallback when the filename alone
# is inconclusive (e.g. journals that publish supplements with opaque names).
_SUPP_CONTENT_PATTERNS = re.compile(
    r"(supplement(?:ary)?\s+(?:material|method|figure|table|note|data|text)|"
    r"supporting\s+information|extended\s+data|"
    r"materials?\s+and\s+methods\s*$)",
    re.IGNORECASE | re.MULTILINE,
)

_SUPP_TYPE_PATTERNS = {
    "supplementary_methods": re.compile(
        r"(supplement.*method|method.*supplement|supp.*method|extended.*method)", re.I
    ),
    "supplementary_figures": re.compile(
        r"(supplement.*fig|supp.*fig|figure\s*s\d|extended.*fig)", re.I
    ),
    "supplementary_tables": re.compile(
        r"(supplement.*table|supp.*table|table\s*s\d|extended.*table)", re.I
    ),
    "supplementary_data": re.compile(
        r"(supplement.*data|extended.*data|additional.*data|source.*data)", re.I
    ),
    "appendix": re.compile(r"(appendix|additional[\._\-]file)", re.I),
}


# ---------- data structures ----------
class ParsedPage:
    __slots__ = ("page_num", "text", "tables")

    def __init__(self, page_num: int, text: str, tables: list):
        self.page_num = page_num
        self.text = text
        self.tables = tables

    def to_dict(self):
        return {"page_num": self.page_num, "text": self.text, "tables": self.tables}


class ParsedDocument:
    __slots__ = (
        "file_path", "file_name", "role", "sub_type",
        "pages", "full_text", "page_count", "metadata",
    )

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self.role = "unknown"          # main | supplement
        self.sub_type = "general"      # supplementary_methods, etc.
        self.pages: list[ParsedPage] = []
        self.full_text = ""
        self.page_count = 0
        self.metadata: dict = {}

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "role": self.role,
            "sub_type": self.sub_type,
            "page_count": self.page_count,
            "full_text_length": len(self.full_text),
            "metadata": self.metadata,
        }


# ---------- parsing ----------
def parse_pdf(pdf_path: str) -> ParsedDocument:
    """Extract text and tables from a single PDF."""
    doc = ParsedDocument(pdf_path)
    pages = []
    texts = []

    with pdfplumber.open(pdf_path) as pdf:
        doc.page_count = len(pdf.pages)
        doc.metadata = pdf.metadata or {}

        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            # convert table cells to strings
            clean_tables = []
            for tbl in tables:
                clean_tables.append(
                    [[str(cell) if cell is not None else "" for cell in row] for row in tbl]
                )
            parsed_page = ParsedPage(page_num=i + 1, text=text, tables=clean_tables)
            pages.append(parsed_page)
            texts.append(text)

    doc.pages = pages
    doc.full_text = "\n\n".join(texts)
    return doc


def _classify_single(doc: ParsedDocument) -> None:
    """Classify a single document by filename and content heuristics."""
    name = doc.file_name.lower()

    # 1) filename-based detection
    if _SUPP_PATTERNS.search(name):
        doc.role = "supplement"
        for sub_type, pattern in _SUPP_TYPE_PATTERNS.items():
            if pattern.search(name):
                doc.sub_type = sub_type
                return
        # check first‐page content if filename isn't specific
        first_page = doc.pages[0].text[:500].lower() if doc.pages else ""
        for sub_type, pattern in _SUPP_TYPE_PATTERNS.items():
            if pattern.search(first_page):
                doc.sub_type = sub_type
                return
        doc.sub_type = "general_supplement"
        return

    # 2) content-based fallback when filename is opaque
    head_text = ""
    if doc.pages:
        head_text = "\n".join(p.text for p in doc.pages[:2])[:2000]
    if head_text and _SUPP_CONTENT_PATTERNS.search(head_text):
        doc.role = "supplement"
        for sub_type, pattern in _SUPP_TYPE_PATTERNS.items():
            if pattern.search(head_text):
                doc.sub_type = sub_type
                return
        doc.sub_type = "general_supplement"
        return

    # 3) default: treat as main paper
    doc.role = "main"
    doc.sub_type = "main_paper"


def classify_documents(
    pdf_paths: list[str], main_hint: Optional[str] = None
) -> dict:
    """
    Parse and classify multiple PDFs.

    Args:
        pdf_paths: list of PDF file paths
        main_hint: optional filename substring to force as main paper

    Returns:
        {"main": ParsedDocument | None, "supplements": [ParsedDocument, ...]}
    """
    docs = [parse_pdf(p) for p in pdf_paths]

    # if only one doc, it's the main paper
    if len(docs) == 1:
        docs[0].role = "main"
        docs[0].sub_type = "main_paper"
        return {"main": docs[0], "supplements": []}

    # if main_hint provided, use it
    if main_hint:
        hint = main_hint.lower()
        for doc in docs:
            if hint in doc.file_name.lower():
                doc.role = "main"
                doc.sub_type = "main_paper"
            else:
                # Run heuristic classification first; if it still says "main"
                # while we already have a hint-matched main, demote to supplement
                # so multiple inputs with a hint never collapse into multiple mains.
                _classify_single(doc)
        # post-pass: enforce single main when hint matched at least one doc
        hint_mains = [d for d in docs if hint in d.file_name.lower()]
        if hint_mains:
            for doc in docs:
                if doc.role == "main" and hint not in doc.file_name.lower():
                    doc.role = "supplement"
                    if doc.sub_type == "main_paper":
                        doc.sub_type = "general_supplement"
    else:
        for doc in docs:
            _classify_single(doc)

    # if no main detected, pick the longest non-supplement
    mains = [d for d in docs if d.role == "main"]
    supps = [d for d in docs if d.role == "supplement"]
    unknowns = [d for d in docs if d.role == "unknown"]

    if not mains:
        # promote the longest unknown or supplement
        candidates = unknowns if unknowns else supps
        if candidates:
            candidates.sort(key=lambda d: d.page_count, reverse=True)
            candidates[0].role = "main"
            candidates[0].sub_type = "main_paper"
            mains = [candidates[0]]
            supps = [d for d in docs if d.role == "supplement"]

    # remaining unknowns become supplements
    for doc in docs:
        if doc.role == "unknown":
            doc.role = "supplement"
            doc.sub_type = "general_supplement"
            supps.append(doc)

    main_doc = mains[0] if mains else None

    # sort supplements by priority: methods > tables > figures > data > others
    priority_order = {
        "supplementary_methods": 0,
        "supplementary_tables": 1,
        "supplementary_figures": 2,
        "supplementary_data": 3,
        "appendix": 4,
        "general_supplement": 5,
    }
    supps.sort(key=lambda d: priority_order.get(d.sub_type, 9))

    return {"main": main_doc, "supplements": supps}


# ---------- section extraction ----------

# Canonical section names for normalization
_SECTION_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "introduction",
    "related work": "introduction",
    "methods": "methods",
    "methodology": "methods",
    "method": "methods",
    "materials and methods": "methods",
    "material and methods": "methods",
    "experimental": "methods",
    "experiments": "methods",
    "experimental setup": "methods",
    "experimental design": "methods",
    "online methods": "methods",
    "star methods": "methods",
    "results": "results",
    "results and discussion": "results_and_discussion",
    "discussion": "discussion",
    "conclusions": "conclusions",
    "conclusion": "conclusions",
    "data availability": "data_availability",
    "data and code availability": "data_availability",
    "code availability": "code_availability",
    "acknowledgements": "acknowledgements",
    "acknowledgement": "acknowledgements",
    "references": "references",
    "supplementary": "supplementary",
    "funding": "funding",
    "author contributions": "author_contributions",
    "author contribution": "author_contributions",
    "online content": "online_content",
    "competing interests": "competing_interests",
    "additional information": "additional_information",
    "extended data": "extended_data",
    "supplementary information": "supplementary",
    "supporting information": "supplementary",
}

# Primary pattern: strict, header on its own line
_SECTION_HEADERS_STRICT = re.compile(
    r"^\s*(?:\d+[\.\)]\s*)?"
    r"(abstract|introduction|background|related\s+work|"
    r"method(?:s|ology)?|materials?\s+and\s+methods?|"
    r"online\s+methods|star\s+methods|"
    r"experiment(?:s|al)?(?:\s+(?:setup|design))?|"
    r"result(?:s)?(?:\s+and\s+discussion)?|"
    r"discussion|conclusion(?:s)?|"
    r"data(?:\s+and\s+code)?\s+availability|code\s+availability|"
    r"acknowledgement(?:s)?|references|"
    r"supplementary|funding|author\s+contributions?)"
    r"\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Secondary pattern: relaxed, header at line start, may be followed by text on same line
# Uses a lookahead to only match if followed by a newline within 5 chars (short trailing text OK)
_SECTION_HEADERS_RELAXED = re.compile(
    r"^\s*(?:\d+[\.\)]\s*)?"
    r"(abstract|introduction|background|"
    r"method(?:s|ology)?|materials?\s+and\s+methods?|"
    r"online\s+methods|star\s+methods|"
    r"result(?:s)?(?:\s+and\s+discussion)?|"
    r"discussion|conclusion(?:s)?)"
    r"[:\.\s]{0,5}$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_section_name(raw: str) -> str:
    """Normalize a raw section header to a canonical name."""
    key = raw.strip().lower()
    # Try exact match first
    if key in _SECTION_ALIASES:
        return _SECTION_ALIASES[key]
    # Try prefix match
    for alias, canonical in _SECTION_ALIASES.items():
        if key.startswith(alias) or alias.startswith(key):
            return canonical
    return key.replace(" ", "_")


def extract_sections(text: str) -> dict[str, str]:
    """Split paper text into named sections using two-pass header detection.

    Kept as the text-only baseline.  For font-aware + LLM-enhanced extraction
    see :func:`extract_sections_enhanced`.
    """
    return _extract_sections_regex(text)


def _extract_sections_regex(text: str) -> dict[str, str]:
    """Internal: original two-pass regex section splitter."""
    matches = list(_SECTION_HEADERS_STRICT.finditer(text))
    if len(matches) < 3:
        relaxed_matches = list(_SECTION_HEADERS_RELAXED.finditer(text))
        seen_positions = {m.start() for m in matches}
        for rm in relaxed_matches:
            if rm.start() not in seen_positions:
                matches.append(rm)
                seen_positions.add(rm.start())
        matches.sort(key=lambda m: m.start())
    if not matches:
        return {"full_text": text}

    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        raw_name = m.group(1).strip()
        name = _normalize_section_name(raw_name)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            if name in sections:
                sections[name] += "\n\n" + content
            else:
                sections[name] = content
    pre = text[: matches[0].start()].strip()
    if pre:
        sections["preamble"] = pre
    return sections


# ---------- Layer 1: font-aware header detection ----------

# Fonts that are bold but are NOT section headers (article-type labels, author names)
_NON_HEADER_FONTS = {
    "harding-extrabold",  # Nature "Article" page label
    "hardingbold",        # sometimes used for page labels too
}

# Patterns that identify page-number footers or running headers (NOT real sections)
_FOOTER_PATTERNS = re.compile(
    r"^\s*\d+\s*\|\s*\w+.*\.(com|org|edu)|"  # "2 | Nature | www.nature.com"
    r"^\s*\w+\s*\|\s*www\.",                  # "Nature | www.nature.com"
    re.IGNORECASE,
)

# Patterns for figure/table labels that use bold font but are NOT sections
_FIGURE_LABEL_PATTERNS = re.compile(
    r"^(fig(?:ure)?|table|tab|extended\s+data|supplementary|figs?\.?\s*s?\d)",
    re.IGNORECASE,
)


def _is_bold_font(font_name: str) -> bool:
    """Check if a font name indicates bold weight.

    Matches "Bold" as a standalone weight indicator (e.g. "HardingText-Bold",
    "Helvetica-Bold") but NOT "Semibold" or "Extrabold" where "bold" is part
    of a compound weight name.
    """
    fn_lower = font_name.lower()
    if fn_lower in _NON_HEADER_FONTS:
        return False
    # "Semibold" and "Extrabold" are NOT true bold weights
    if "semibold" in fn_lower or "extrabold" in fn_lower or "demibold" in fn_lower:
        return False
    # Match -Bold, _Bold, or standalone Bold (case-sensitive in mixed-case names)
    if re.search(r'(?<=[\-\s_.])Bold', font_name):
        return True
    # Case-insensitive fallback for all-lowercase font names
    if re.search(r'(?:^|[\-_.\s])bold(?:$|[\-_.\s\d])', fn_lower):
        return True
    return False


def extract_bold_headers(pdf_path):
    """
    Detect section headers in a PDF using font-weight metadata.

    Heuristic: lines rendered in a Bold font at a size between body text
    (~7–8 pt) and page-level labels (≥14 pt) are likely section headers.

    Returns:
        list of {"page" (1-indexed), "text", "size", "font"}
    """
    headers: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            chars = page.chars
            if not chars:
                continue
            # group characters by y-position → lines
            line_groups: dict[float, list[dict]] = {}
            for c in chars:
                y = round(c["top"], 0)
                line_groups.setdefault(y, []).append(c)

            for y in sorted(line_groups):
                clist = sorted(line_groups[y], key=lambda x: x["x0"])
                text = "".join(c["text"] for c in clist).strip()
                if not text or len(text) < 4:
                    continue
                font_names = {c.get("fontname", "") for c in clist[: min(10, len(clist))]}
                sizes = {round(c.get("size", 0), 1) for c in clist[: min(10, len(clist))]}
                max_size = max(sizes)

                # Bold font (excludes Semibold/Extrabold via _is_bold_font),
                # section-header size range (≥9 pt)
                is_bold = any(_is_bold_font(fn) for fn in font_names)
                if not is_bold:
                    continue
                if max_size < 9.0 or max_size > 13.0:
                    continue
                # Skip page footers / running headers
                if _FOOTER_PATTERNS.search(text):
                    continue
                # Skip figure/table labels (e.g. "Figure 1", "Table S1")
                if _FIGURE_LABEL_PATTERNS.search(text):
                    continue
                # Skip very long "headers" (> 80 chars is almost certainly body text)
                if len(text) > 80:
                    continue
                # Skip short bold labels in the lower half of the page —
                # these are likely figure/table annotations, not section headers.
                page_height = page.height or 800
                if len(text) < 25 and y > page_height * 0.5:
                    continue
                headers.append({
                    "page": i + 1,
                    "text": text,
                    "size": max_size,
                    "font": next(iter(font_names), ""),
                })
    return headers


# ---------- Layer 2: LLM section identification ----------

_SECTION_LLM_PROMPT = """You are a precise document structure analyser. \
Given the text of an academic paper, identify ALL major section boundaries.

For each section, return:
- "start_offset": the character offset (0-indexed) where the section CONTENT begins \
  (right after the header text, at the first character of the body paragraph).
- "name": the section header text as it appears.
- "canonical": one of: introduction, results, results_and_discussion, discussion, \
  methods, conclusions, data_availability, code_availability, references, \
  acknowledgements, author_contributions, online_content, or a descriptive sub-heading \
  (use snake_case for descriptive names, e.g. "dendritic_cell_notch_signalling").

Rules:
1. If the paper has no explicit "Introduction" header but starts with body text after \
   the abstract, set introduction start_offset to the first character of that body text.
2. Descriptive sub-headings (e.g. Nature-style "A cellular atlas of ...") should be \
   classified as results sub-sections.
3. "Online content" / "Online Methods" / "Data availability" / "Code availability" \
   are standard end-matter sections.
4. Do NOT invent sections that are not supported by the text.
5. Output MUST be a valid JSON array of objects, no markdown fences.

Text (first 30000 chars):
{text}"""


def _identify_sections_llm(text: str, llm_config: dict) -> list[dict]:
    """Call LLM to identify section boundaries. Returns list of boundary dicts."""
    from openai import OpenAI

    client = OpenAI(api_key=llm_config["api_key"], base_url=llm_config["base_url"])
    truncated = text[:30000]

    try:
        resp = client.chat.completions.create(
            model=llm_config["model"],
            messages=[
                {"role": "system", "content": "You are a precise document structure analyser. Output only valid JSON."},
                {"role": "user", "content": _SECTION_LLM_PROMPT.format(text=truncated)},
            ],
            temperature=0,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        print(f"  [warn] LLM section identification failed: {e}")
        return []


# ---------- Layer 3: validation ----------

def _validate_boundaries(text: str, boundaries: list[dict]) -> list[dict]:
    """
    Validate LLM-identified boundaries against the original text.
    Check that the area around start_offset contains text consistent with
    the claimed section name.
    """
    validated: list[dict] = []
    for b in boundaries:
        offset = b.get("start_offset")
        if not isinstance(offset, int) or offset < 0 or offset >= len(text):
            continue
        # Look at 200 chars before the offset for a header-like string
        window_start = max(0, offset - 300)
        window = text[window_start:offset].lower()
        name_lower = b.get("name", "").lower()
        canonical = b.get("canonical", "")

        # Check: does the header text appear near the offset?
        confident = False
        if name_lower and name_lower[:10] in window:
            confident = True
        elif canonical in _SECTION_ALIASES:
            # For canonical sections, check the canonical name in the window
            if canonical.replace("_", " ")[:6] in window:
                confident = True

        entry = {
            "name": b.get("name", canonical),
            "canonical": canonical,
            "start_offset": offset,
            "confidence": "high" if confident else "low",
        }
        validated.append(entry)
    return validated


# ---------- orchestrator: three-layer section extraction ----------

def extract_sections_enhanced(
    text: str,
    pdf_path: Optional[str] = None,
    llm_config: Optional[dict] = None,
    doc: Optional["ParsedDocument"] = None,
) -> dict[str, str]:
    """
    Three-layer section extraction:

    Layer 1 (font-aware):  Use pdfplumber character metadata to detect bold
        sub-headings (Nature/Science/Cell style). Zero cost, high accuracy
        for well-typeset PDFs.
    Layer 2 (regex):       Traditional two-pass regex for standard headers
        (Discussion, Methods, etc).
    Layer 3 (LLM fallback): If Layers 1+2 yield < 3 sections, ask LLM to
        identify section boundaries by character offset. Expensive but robust.

    Args:
        text: full paper text
        pdf_path: path to the original PDF (for font detection)
        llm_config: LLM API config dict (for LLM fallback)
        doc: ParsedDocument with per-page text (for offset mapping)
    """
    # --- Layer 2: regex (always run — cheap and reliable for standard names) ---
    regex_sections = _extract_sections_regex(text)
    regex_names = set(regex_sections.keys()) - {"preamble", "full_text"}

    # --- Layer 1: font-aware detection ---
    font_headers: list[dict] = []
    if pdf_path:
        try:
            font_headers = extract_bold_headers(pdf_path)
        except Exception:
            font_headers = []

    # Map font-detected headers to text offsets
    font_entries: list[dict] = []
    if font_headers and doc:
        page_offsets = _build_page_offsets(doc)
        for h in font_headers:
            page_idx = h["page"] - 1
            if page_idx >= len(doc.pages):
                continue
            page_text = doc.pages[page_idx].text
            header_text = h["text"]

            # Search for header text within the page region of full_text.
            # Use progressively shorter keys to handle two-column reordering
            # where extract_text() order may differ from visual char order.
            found_offset = -1
            search_region_start = page_offsets[page_idx]
            search_region_end = (
                page_offsets[page_idx + 1]
                if page_idx + 1 < len(page_offsets)
                else len(text)
            )
            search_region = text[search_region_start:search_region_end]

            # Try full header text first, with progressively shorter keys
            # to handle two-column reordering.  Minimum 5 chars covers
            # short canonical headers like "Methods".
            for key_len in (len(header_text), 40, 25, 15, 7, 5):
                search_key = header_text[:key_len].strip()
                if len(search_key) < 5:
                    break
                pos = search_region.find(search_key)
                if pos == -1:
                    # Try case-insensitive
                    pos = search_region.lower().find(search_key.lower())
                if pos >= 0:
                    found_offset = search_region_start + pos
                    break

            if found_offset < 0:
                continue

            # Content starts after the header text.  For two-column layouts,
            # the header and body text may be on the same line; use the
            # header text length to skip past the header, then advance to
            # the next newline boundary.
            content_start = found_offset + len(header_text)
            end_of_line = text.find("\n", content_start)
            if end_of_line == -1 or (end_of_line - content_start) > 2:
                content_offset = content_start
            else:
                content_offset = end_of_line + 1

            font_entries.append({
                "name": header_text,
                "canonical": _normalize_section_name(header_text),
                "header_offset": found_offset,
                "content_offset": content_offset,
                "source": "font",
            })

    # --- Merge Layer 1 + Layer 2 ---
    # Strategy: if regex already found enough standard sections (≥ 3),
    # font detection is redundant for standard headers and risks adding
    # false positives (figure labels, annotations).  Only use font
    # entries to *supplement* — add descriptive sub-headings that
    # regex cannot find.
    regex_found_enough = len(regex_names) >= 3

    # Known standard section canonicals (from _SECTION_ALIASES values)
    _known_canonicals = set(_SECTION_ALIASES.values())

    merged_entries: list[dict] = []
    seen_canonicals: set[str] = set()

    for entry in font_entries:
        # When regex is sufficient, only keep font entries that add NEW
        # information — either descriptive sub-headings or standard
        # sections that regex MISSED.
        if regex_found_enough:
            canon = entry["canonical"]
            if canon in regex_names:
                continue  # already covered by regex — skip duplicate
        else:
            # Regex found few sections — font layer is the primary source.
            # Accept entries whose canonical name is a known standard section
            # (handles 1-2 word back matter like "References", "Methods",
            # "Acknowledgements", "Author contributions").  For descriptive
            # sub-headings (unknown canonical), require ≥ 3 words and an
            # uppercase first word to filter out figure labels.
            canon = entry["canonical"]
            if canon not in _known_canonicals:
                name = entry["name"]
                words = name.split()
                if len(words) < 3:
                    continue
                if not words[0][0].isupper():
                    continue
        merged_entries.append(entry)
        seen_canonicals.add(entry["canonical"])

    # Add regex sections not covered by font detection
    regex_matches = list(_SECTION_HEADERS_STRICT.finditer(text))
    if len(regex_matches) < 3:
        relaxed = list(_SECTION_HEADERS_RELAXED.finditer(text))
        seen_pos = {m.start() for m in regex_matches}
        for rm in relaxed:
            if rm.start() not in seen_pos:
                regex_matches.append(rm)
                seen_pos.add(rm.start())
        regex_matches.sort(key=lambda m: m.start())

    for m in regex_matches:
        canon = _normalize_section_name(m.group(1).strip())
        if canon not in seen_canonicals:
            merged_entries.append({
                "name": m.group(1).strip(),
                "canonical": canon,
                "header_offset": m.start(),
                "content_offset": m.end(),
                "source": "regex",
            })
            seen_canonicals.add(canon)

    merged_entries.sort(key=lambda e: e["header_offset"])

    # Slice text
    sections: dict[str, str] = {}
    _font_canonicals: set[str] = set()
    for i, entry in enumerate(merged_entries):
        start = entry["content_offset"]
        end = merged_entries[i + 1]["header_offset"] if i + 1 < len(merged_entries) else len(text)
        content = text[start:end].strip()
        canon = entry["canonical"]

        # Skip font-detected sections with negligible content — they are
        # likely inline mentions rather than real headers.  Let the regex
        # layer or a later run pick them up properly.
        if entry["source"] == "font" and len(content) < 20:
            continue

        if content:
            if canon in sections:
                sections[canon] += "\n\n" + content
            else:
                sections[canon] = content
            if entry["source"] == "font":
                _font_canonicals.add(canon)

    # Preamble
    if merged_entries:
        pre = text[: merged_entries[0]["header_offset"]].strip()
        if pre:
            sections["preamble"] = pre

    # --- Layer 3: LLM fallback if still too few ---
    unique_real = set(sections.keys()) - {"preamble", "full_text"}
    if len(unique_real) < 3 and llm_config:
        llm_boundaries = _identify_sections_llm(text, llm_config)
        if llm_boundaries:
            validated = _validate_boundaries(text, llm_boundaries)
            # Only use LLM results if at least half are high-confidence
            high_conf = [v for v in validated if v["confidence"] == "high"]
            if len(high_conf) >= 2:
                validated.sort(key=lambda v: v["start_offset"])
                sections = {}
                for i, v in enumerate(validated):
                    start = v["start_offset"]
                    end = validated[i + 1]["start_offset"] if i + 1 < len(validated) else len(text)
                    content = text[start:end].strip()
                    canon = v.get("canonical", v["name"])
                    if content:
                        if canon in sections:
                            sections[canon] += "\n\n" + content
                        else:
                            sections[canon] = content
                if validated:
                    pre = text[: validated[0]["start_offset"]].strip()
                    if pre:
                        sections["preamble"] = pre

    if not sections:
        return {"full_text": text}
    return sections


def _build_page_offsets(doc: "ParsedDocument") -> list[int]:
    """Build character offsets for the start of each page in full_text."""
    offsets: list[int] = []
    running = 0
    for page in doc.pages:
        offsets.append(running)
        running += len(page.text) + 2  # +2 for the \n\n separator
    return offsets


# ---------- figure / table ID extraction ----------

# Matches: Figure 1, Fig. 2, Fig 3a, Table 1, Table S1, Extended Data Fig. 1,
# Supplementary Figure 1, Supplementary Table 2, etc.
_FIGURE_TABLE_ID = re.compile(
    r"(?:Extended\s+Data\s+)?"
    r"(?:Supplementary\s+)?"
    r"(?:Figure|Fig\.?|Table|Tab\.?)"
    r"\s*([Ss]?\d+[a-z]?(?:\s*[-–]\s*[a-z])?)",
    re.IGNORECASE,
)


def extract_figure_table_ids(text: str) -> list[str]:
    """
    Scan full text and return a deduplicated, sorted list of all figure/table
    identifiers mentioned (e.g., ['Figure 1', 'Figure 2a', 'Table S1']).
    """
    raw_matches = _FIGURE_TABLE_ID.finditer(text)
    seen = {}  # canonical -> display form
    for m in raw_matches:
        full = m.group(0).strip()
        # Normalize to canonical form for dedup
        canonical = re.sub(r"\s+", " ", full).strip()
        # Normalize Fig./Fig to Figure, Tab./Tab to Table
        canonical = re.sub(r"\bFig\.?\b", "Figure", canonical, flags=re.I)
        canonical = re.sub(r"\bTab\.?\b", "Table", canonical, flags=re.I)
        # Remove trailing letter ranges for dedup (Figure 2a-c -> Figure 2)
        base_key = re.sub(r"([Ss]?\d+)[a-z](?:\s*[-–]\s*[a-z])?$", r"\1", canonical)
        if base_key not in seen:
            seen[base_key] = canonical

    # Sort: main figures first, then tables, then supplementary/extended
    def sort_key(item):
        k = item.lower()
        prefix = 0
        if "extended" in k:
            prefix = 2
        elif "supplementary" in k:
            prefix = 3
        elif "table" in k:
            prefix = 1
        # extract number
        num_match = re.search(r"(\d+)", k)
        num = int(num_match.group(1)) if num_match else 99
        return (prefix, num)

    return sorted(seen.values(), key=sort_key)
