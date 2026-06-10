"""
Hlokk - Report Generator
Renders structured understanding data into interactive HTML using Jinja2 templates.
"""
from pathlib import Path
from typing import Optional

from jinja2 import Environment, BaseLoader


_TEMPLATE_PATH = Path(__file__).parent / "report_template.html"


def generate_html_report(
    output_data: dict,
    figure_data: Optional[dict] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Render an interactive HTML report from Hlokk output data.

    Args:
        output_data: the full Hlokk JSON output (understanding + metadata)
        figure_data: output from figure_extractor.extract_figures_for_report()
        output_path: if provided, write HTML to this path

    Returns:
        rendered HTML string
    """
    template_str = _TEMPLATE_PATH.read_text(encoding="utf-8")
    env = Environment(loader=BaseLoader(), autoescape=False)
    # register custom filters
    env.filters["default"] = lambda v, d="": v if v else d
    template = env.from_string(template_str)

    understanding = output_data.get("understanding", {})
    profile = output_data.get("user_profile_snapshot", {})

    # prepare figure-page associations
    page_images = {}
    figure_page_map = {}
    if figure_data:
        page_images = figure_data.get("page_images", {})
        figure_page_map = figure_data.get("figure_page_map", {})

    # enrich figure_arguments with page image availability
    fig_args = understanding.get("figure_arguments", [])
    for fig in fig_args:
        fig["_has_image"] = False
        fig["_page_num"] = None
        loc = fig.get("evidence_location", "")
        import re
        match = re.search(r"(?:page|p\.?)\s*(\d+)", loc, re.IGNORECASE)
        if match:
            pn = int(match.group(1))
            fig["_page_num"] = pn
            if pn in page_images:
                fig["_has_image"] = True

    html = template.render(
        # metadata
        hlokk_version=output_data.get("hlokk_version", "2.2.0"),
        generated_at=output_data.get("generated_at", ""),
        paper_id=output_data.get("paper_id", ""),
        analysis_mode=output_data.get("analysis_mode", "overview"),
        doc_type=output_data.get("doc_type", "article"),
        input_files=output_data.get("input_files", {}),
        main_sections=output_data.get("main_sections", []),
        detected_figure_table_ids=output_data.get("detected_figure_table_ids", []),
        rag_stats=output_data.get("rag_stats", {}),
        coverage_gaps=output_data.get("coverage_gaps", {}),
        # understanding
        u=understanding,
        # reading cipher
        rc=understanding.get("reading_cipher", {}),
        # figure arguments
        fig_args=fig_args,
        # user profile
        profile=profile,
        # figure images
        page_images=page_images,
        figure_page_map=figure_page_map,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")

    return html
