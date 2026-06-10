"""
Hlokk - Figure Extractor
Renders PDF pages containing figures/tables as images for HTML report embedding.
Uses PyMuPDF page rendering (reliable for both vector and raster figures).
"""
import re
import base64
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


def parse_page_refs(figure_arguments: list[dict]) -> dict[int, list[str]]:
    """
    Parse page numbers from figure_arguments evidence_location.
    Returns {page_num: [figure_id, ...]}.
    """
    page_to_figs: dict[int, list[str]] = {}
    for fig in figure_arguments:
        loc = fig.get("evidence_location", "")
        fig_id = fig.get("figure_id", "Unknown")
        # match patterns: "page 5", "Page 5", "p. 5", "p5", "Page5"
        matches = re.findall(r"(?:page|p\.?)\s*(\d+)", loc, re.IGNORECASE)
        for m in matches:
            pn = int(m)
            page_to_figs.setdefault(pn, []).append(fig_id)
    return page_to_figs


def render_pages(
    pdf_path: str,
    page_numbers: list[int],
    dpi: int = 150,
) -> dict[int, dict]:
    """
    Render specific PDF pages as base64 PNG images.

    Args:
        pdf_path: path to the PDF file
        page_numbers: 1-indexed page numbers to render
        dpi: rendering resolution (150 is a good balance of quality/size)

    Returns:
        {page_num: {"base64": str, "width": int, "height": int, "size_kb": float}}
    """
    doc = fitz.open(pdf_path)
    result = {}

    for pn in page_numbers:
        idx = pn - 1  # fitz uses 0-indexed pages
        if idx < 0 or idx >= len(doc):
            continue
        page = doc[idx]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("ascii")
        result[pn] = {
            "base64": b64,
            "width": pix.width,
            "height": pix.height,
            "size_kb": round(len(img_bytes) / 1024, 1),
        }

    doc.close()
    return result


def extract_figures_for_report(
    pdf_path: str,
    figure_arguments: list[dict],
    dpi: int = 150,
) -> dict:
    """
    High-level function: parse figure page references and render those pages.

    Returns:
        {
            "page_images": {page_num: {"base64": ..., "width": ..., "height": ...}},
            "figure_page_map": {page_num: [fig_id, ...]},
            "total_size_kb": float,
        }
    """
    page_map = parse_page_refs(figure_arguments)

    if not page_map:
        return {"page_images": {}, "figure_page_map": {}, "total_size_kb": 0}

    page_images = render_pages(pdf_path, sorted(page_map.keys()), dpi=dpi)
    total_kb = sum(img["size_kb"] for img in page_images.values())

    return {
        "page_images": page_images,
        "figure_page_map": page_map,
        "total_size_kb": round(total_kb, 1),
    }
