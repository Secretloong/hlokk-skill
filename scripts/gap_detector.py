"""
Hlokk - Gap Detector v2.2
Post-LLM coverage analysis: detects sections, figures, tables and methods
that the LLM missed or insufficiently covered.
Supports multi-document-type awareness (article, review, benchmark, commentary).
"""
import re
from typing import Optional


def _extract_referenced_sections(understanding: dict) -> set[str]:
    """Collect all section names referenced in evidence_location fields."""
    sections = set()
    # Walk all evidence_location strings in the understanding dict
    _walk_evidence_locations(understanding, sections)
    return sections


def _walk_evidence_locations(obj, sections: set):
    """Recursively find evidence_location strings and extract section names."""
    section_pattern = re.compile(
        r"(?:section|sect\.?)[:\s]*([A-Za-z][A-Za-z\s&]+?)(?:\s*[-–/]|\s*$|,|\))",
        re.IGNORECASE,
    )
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "evidence_location" and isinstance(value, str):
                # Extract section references
                for m in section_pattern.finditer(value):
                    sections.add(m.group(1).strip().lower())
                # Also pick up inline section names
                for sname in [
                    "abstract", "introduction", "methods", "results",
                    "discussion", "conclusions", "supplementary",
                ]:
                    if sname in value.lower():
                        sections.add(sname)
            else:
                _walk_evidence_locations(value, sections)
    elif isinstance(obj, list):
        for item in obj:
            _walk_evidence_locations(item, sections)


def _extract_analyzed_figure_ids(understanding: dict) -> set[str]:
    """Get figure/table IDs that the LLM actually analyzed."""
    ids = set()
    for fig in understanding.get("figure_arguments", []):
        fid = fig.get("figure_id", "")
        if fid:
            # Normalize
            canonical = re.sub(r"\bFig\.?\b", "Figure", fid, flags=re.I)
            canonical = re.sub(r"\bTab\.?\b", "Table", canonical, flags=re.I)
            canonical = re.sub(r"\s+", " ", canonical).strip()
            # Extract base (Figure 1, Table 2, ignore sub-panel letters)
            base = re.sub(r"([Ss]?\d+)[a-z](?:\s*[-–]\s*[a-z])?$", r"\1", canonical)
            ids.add(base)
    return ids


def _normalize_fig_id_for_compare(fid: str) -> str:
    """Normalize a figure/table ID for comparison."""
    canonical = re.sub(r"\bFig\.?\b", "Figure", fid, flags=re.I)
    canonical = re.sub(r"\bTab\.?\b", "Table", canonical, flags=re.I)
    canonical = re.sub(r"\s+", " ", canonical).strip()
    base = re.sub(r"([Ss]?\d+)[a-z](?:\s*[-–]\s*[a-z])?$", r"\1", canonical)
    return base


def _get_coverage_config(doc_type: str) -> dict:
    """Return coverage check configuration for each document type."""
    configs = {
        "article": {
            "weights": {"sections": 0.30, "figures": 0.30, "methods": 0.20, "fields": 0.20},
            "foundation_fields": ["key_findings", "reading_cipher", "concepts", "field_positioning"],
            "deep_fields": [
                "core_methods", "figure_arguments",
                "literature_report", "research_recommendations", "relevance_to_user",
            ],
        },
        "review": {
            "weights": {"sections": 0.15, "figures": 0.20, "methods": 0.10, "fields": 0.55},
            "foundation_fields": ["key_findings", "reading_cipher", "concepts", "field_positioning"],
            "deep_fields": [
                "thematic_threads", "figure_arguments", "consensus_and_debate",
                "gaps_and_future", "methods_landscape", "literature_report", "relevance_to_user",
            ],
        },
        "benchmark": {
            "weights": {"sections": 0.15, "figures": 0.35, "methods": 0.20, "fields": 0.30},
            "foundation_fields": ["key_findings", "reading_cipher", "concepts", "field_positioning"],
            "deep_fields": [
                "benchmark_design", "evaluation_threads", "figure_arguments",
                "rankings_and_recommendations", "methodological_insights",
                "literature_report", "relevance_to_user",
            ],
        },
        "commentary": {
            "weights": {"sections": 0.10, "figures": 0.10, "methods": 0.05, "fields": 0.75},
            "foundation_fields": ["key_findings", "reading_cipher", "concepts", "field_positioning"],
            "deep_fields": [
                "commentary_structure", "context_and_stakes",
                "literature_report", "relevance_to_user",
            ],
        },
        "atlas": {
            "weights": {"sections": 0.15, "figures": 0.25, "methods": 0.25, "fields": 0.35},
            "foundation_fields": ["key_findings", "reading_cipher", "concepts", "field_positioning"],
            "deep_fields": [
                "resource_design", "annotation_assessment", "data_accessibility",
                "figure_arguments", "initial_biological_findings",
                "literature_report", "relevance_to_user",
            ],
        },
    }
    return configs.get(doc_type, configs["article"])


def _assess_methods_by_type(understanding: dict, doc_type: str) -> dict:
    """Assess methods/evaluation completeness based on document type."""
    if doc_type == "article":
        # Original logic: check version, parameters, supports_findings
        core_methods = understanding.get("core_methods", [])
        total_methods = len(core_methods)
        if total_methods == 0:
            return {"total": 0, "score": 1.0, "details": {}}
        missing_version = sum(1 for m in core_methods if not m.get("version"))
        missing_params = sum(
            1 for m in core_methods
            if not m.get("parameters") or m.get("parameters") == {}
        )
        missing_mapping = sum(
            1 for m in core_methods
            if not m.get("supports_findings") and not m.get("supports_figures")
        )
        score = 1.0 - (
            0.3 * missing_version / total_methods
            + 0.4 * missing_params / total_methods
            + 0.3 * missing_mapping / total_methods
        )
        return {
            "total": total_methods,
            "missing_version": missing_version,
            "missing_parameters": missing_params,
            "missing_result_mapping": missing_mapping,
            "score": round(score, 2),
        }

    elif doc_type == "review":
        # Review: check methods_landscape coverage
        landscape = understanding.get("methods_landscape", [])
        total = len(landscape)
        # At least 3 method families is considered complete
        score = min(1.0, total / 3) if total > 0 else 0.5  # No landscape is acceptable for some reviews
        return {"total": total, "score": round(score, 2), "type": "methods_landscape"}

    elif doc_type == "benchmark":
        # Benchmark: check benchmark_design completeness
        design = understanding.get("benchmark_design", {})
        completeness_checks = [
            bool(design.get("methods_evaluated")),
            bool(design.get("datasets")),
            bool(design.get("metrics")),
            bool(design.get("evaluation_protocol")),
            bool(design.get("ground_truth")),
        ]
        score = sum(completeness_checks) / len(completeness_checks)
        return {
            "total": len(design.get("methods_evaluated", [])),
            "datasets_count": len(design.get("datasets", [])),
            "metrics_count": len(design.get("metrics", [])),
            "score": round(score, 2),
            "type": "benchmark_design",
        }

    elif doc_type == "atlas":
        # Atlas/Dataset: check resource_design + annotation + accessibility completeness
        resource = understanding.get("resource_design", {})
        annotation = understanding.get("annotation_assessment", {})
        access = understanding.get("data_accessibility", {})
        completeness_checks = [
            bool(resource.get("sample_design")),
            bool(resource.get("technology_platform")),
            bool(resource.get("preprocessing_pipeline")),
            bool(annotation.get("annotation_strategy")),
            bool(annotation.get("validation_method")),
            bool(access.get("repositories")),
            bool(access.get("data_formats")),
        ]
        score = sum(completeness_checks) / len(completeness_checks)
        return {
            "total": len(completeness_checks),
            "fulfilled": sum(completeness_checks),
            "score": round(score, 2),
            "type": "resource_design",
        }

    else:  # commentary
        # Commentary: methods not important
        return {"total": 0, "score": 1.0, "type": "not_applicable"}


def detect_coverage_gaps(
    understanding: dict,
    detected_sections: list[str],
    detected_figure_ids: list[str],
    rag_stats: Optional[dict] = None,
    doc_type: str = "article",
) -> dict:
    """
    Analyze coverage gaps between what the paper contains and what the LLM reported.

    Args:
        understanding: the LLM's structured output dict
        detected_sections: section names extracted by pdf_parser.extract_sections()
        detected_figure_ids: figure/table IDs from pdf_parser.extract_figure_table_ids()
        rag_stats: optional RAG retrieval stats
        doc_type: document type for type-aware scoring ("article"|"review"|"benchmark"|"commentary")

    Returns:
        Coverage gap report dict with scores and missing items.
    """
    config = _get_coverage_config(doc_type)

    # --- 1. Section coverage ---
    # Canonical detected sections (skip preamble, references, acknowledgements, funding)
    content_sections = {
        s for s in detected_sections
        if s not in ("preamble", "references", "acknowledgements", "funding",
                      "author_contributions", "supplementary")
    }
    referenced_sections = _extract_referenced_sections(understanding)

    # Also check: do literature_report / reading_cipher reference major sections implicitly?
    lit_report = understanding.get("literature_report", {})
    if lit_report:
        # If literature_report has results_narrative, the LLM covered results
        if lit_report.get("results_narrative") and len(lit_report["results_narrative"]) > 50:
            referenced_sections.add("results")
        if lit_report.get("literature_review") and len(lit_report["literature_review"]) > 50:
            referenced_sections.add("introduction")
        if lit_report.get("methods_assessment") and len(lit_report["methods_assessment"]) > 50:
            referenced_sections.add("methods")
        if lit_report.get("topic_summary") and len(lit_report["topic_summary"]) > 50:
            referenced_sections.add("abstract")

    # Normalize for comparison
    normalized_detected = set()
    for s in content_sections:
        ns = s.lower().replace("_", " ").strip()
        normalized_detected.add(ns)

    normalized_referenced = set()
    for s in referenced_sections:
        ns = s.lower().replace("_", " ").strip()
        normalized_referenced.add(ns)

    # Fuzzy match: "results and discussion" covers both "results" and "discussion"
    if "results and discussion" in normalized_detected:
        normalized_detected.add("results")
        normalized_detected.add("discussion")
    if "results and discussion" in normalized_referenced:
        normalized_referenced.add("results")
        normalized_referenced.add("discussion")

    missing_sections = sorted(normalized_detected - normalized_referenced)
    section_score = (
        1.0 - len(missing_sections) / max(len(normalized_detected), 1)
        if normalized_detected else 1.0
    )

    # --- 2. Figure/Table coverage ---
    detected_fig_set = {_normalize_fig_id_for_compare(f) for f in detected_figure_ids}
    analyzed_fig_set = _extract_analyzed_figure_ids(understanding)

    # Only count main figures/tables (not supplementary) for coverage score
    main_figs = {f for f in detected_fig_set if "supplementary" not in f.lower() and "extended" not in f.lower()}
    missing_figures = sorted(main_figs - analyzed_fig_set)
    all_missing_figures = sorted(detected_fig_set - analyzed_fig_set)

    figure_score = (
        1.0 - len(missing_figures) / max(len(main_figs), 1)
        if main_figs else 1.0
    )

    # --- 3. Methods completeness (type-aware) ---
    methods_report = _assess_methods_by_type(understanding, doc_type)
    methods_score = methods_report["score"]

    # --- 4. Key output fields completeness (type-aware) ---
    # In overview mode (no _deep_pass), only Foundation fields are expected.
    is_deep = bool(understanding.get("_deep_pass"))
    foundation_fields = config["foundation_fields"]
    deep_only_fields = config["deep_fields"]
    expected_fields = foundation_fields + deep_only_fields if is_deep else foundation_fields
    present_fields = []
    empty_fields = []
    for field in expected_fields:
        val = understanding.get(field)
        if val and val != {} and val != []:
            present_fields.append(field)
        else:
            empty_fields.append(field)

    fields_score = len(present_fields) / max(len(expected_fields), 1)

    # --- 5. Overall score (type-weighted) ---
    weights = config["weights"]
    overall = (
        weights["sections"] * section_score
        + weights["figures"] * figure_score
        + weights["methods"] * methods_score
        + weights["fields"] * fields_score
    )

    return {
        "doc_type": doc_type,
        "sections_coverage": {
            "detected": sorted(normalized_detected),
            "referenced_by_llm": sorted(normalized_referenced),
            "missing": missing_sections,
            "score": round(section_score, 2),
        },
        "figures_coverage": {
            "detected_in_text": sorted(detected_fig_set),
            "analyzed_by_llm": sorted(analyzed_fig_set),
            "missing_main": missing_figures,
            "missing_all": all_missing_figures,
            "score": round(figure_score, 2),
        },
        "methods_completeness": methods_report,
        "output_fields": {
            "present": present_fields,
            "empty": empty_fields,
            "score": round(fields_score, 2),
        },
        "overall_score": round(overall, 2),
    }
