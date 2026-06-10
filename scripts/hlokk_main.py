#!/usr/bin/env python3
"""
Hlokk - Main Orchestrator v2.1
Academic paper interpretation pipeline with dual-call LLM architecture.

Usage:
    python hlokk_main.py --pdfs paper.pdf supp.pdf --workspace /path/to/workspace
    python hlokk_main.py --pdfs paper.pdf --workspace . --focus "spatial methods"
    python hlokk_main.py --pdfs paper.pdf --workspace . --skip-rag
    python hlokk_main.py --pdfs paper.pdf --workspace . --with-figures
    python hlokk_main.py --pdfs paper.pdf --workspace . --mode detailed
    python hlokk_main.py --pdfs paper.pdf --workspace . --mode detailed --ingest --zotero-key ABC123
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

# allow imports from the same directory
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel

from utils import load_yaml, save_json, get_timestamp, paper_id_from_paths, ensure_dir
from pdf_parser import classify_documents, extract_sections, extract_sections_enhanced, extract_figure_table_ids
from text_chunker import chunk_text, chunk_document_sections
from vector_store import (
    init_store, store_chunks, retrieve_relevant, build_queries_from_profile,
)
from llm_client import understand_paper, _resolve_deep_type
from figure_extractor import extract_figures_for_report
from report_generator import generate_html_report
from gap_detector import detect_coverage_gaps

console = Console()


def _enrich_paper_urls(understanding: dict, zotero_key: str = None,
                       paper_url_override: str = None) -> dict:
    """
    Enrich the understanding dict with clickable URLs for the HTML report header.

    Adds:
      - paper_url:   the article landing page (clickable title target)
      - journal_url: publisher/preprint server page for the article

    Priority:
      1. --paper-url CLI override (e.g. from Zotero MCP — most reliable)
      2. Zotero public API (works for public / group libraries)
      3. DOI URL (browsers resolve doi.org redirects natively)
    """
    doi_raw = understanding.get("doi") or ""
    # Normalise DOI into a full URL
    if doi_raw:
        doi_url = doi_raw if doi_raw.startswith("http") else f"https://doi.org/{doi_raw}"
    else:
        doi_url = None

    paper_url = paper_url_override or doi_url
    journal_url = None

    # ---- Try Zotero public API for richer metadata ----
    if zotero_key and not paper_url_override:
        try:
            zotero_api = f"https://api.zotero.org/items/{zotero_key}?format=json"
            req = urllib.request.Request(zotero_api, headers={"User-Agent": "Hlokk/2.2"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            item_data = data.get("data", {})

            zot_url = item_data.get("url", "")
            zot_doi = item_data.get("DOI", "")

            if zot_url:
                paper_url = zot_url
            elif zot_doi and not paper_url:
                paper_url = f"https://doi.org/{zot_doi}"
        except Exception:
            pass

    # ---- Resolve DOI redirect to get publisher landing page ----
    # Note: many publishers (bioRxiv, Nature, etc.) block programmatic HEAD
    # requests with Cloudflare / 403. Browsers resolve DOI redirects natively,
    # so the DOI URL is a reliable default for clickable links.
    # For a direct publisher URL, pass --paper-url from the orchestrator.

    # ---- Derive journal / publisher page ----
    if paper_url:
        journal_url = paper_url  # article landing page = journal page

    if not journal_url and paper_url:
        journal_url = paper_url

    understanding["paper_url"] = paper_url
    understanding["journal_url"] = journal_url
    return understanding


def run_pipeline(args: argparse.Namespace) -> dict:
    workspace = Path(args.workspace)
    config_path = workspace / "config.yaml"
    profile_path = workspace / "profile.yaml"

    # ------ validate prerequisites ------
    if not config_path.exists():
        console.print("[red]ERROR:[/red] config.yaml not found in workspace.")
        console.print("Copy the template:  cp <skill_dir>/config_template.yaml config.yaml")
        sys.exit(1)
    if not profile_path.exists():
        console.print("[red]ERROR:[/red] profile.yaml not found in workspace.")
        console.print("Copy the template:  cp <skill_dir>/profile_template.yaml profile.yaml")
        sys.exit(1)

    config = load_yaml(str(config_path))
    profile = load_yaml(str(profile_path))

    # ------ env var overrides (highest priority) ------
    if os.environ.get("EMBEDDING_API_KEY"):
        config.setdefault("embedding", {})["api_key"] = os.environ["EMBEDDING_API_KEY"]
    if os.environ.get("EMBEDDING_BASE_URL"):
        config.setdefault("embedding", {})["base_url"] = os.environ["EMBEDDING_BASE_URL"]
    if os.environ.get("EMBEDDING_MODEL"):
        config.setdefault("embedding", {})["model"] = os.environ["EMBEDDING_MODEL"]
    if os.environ.get("HLOKK_LLM_KEY"):
        config.setdefault("llm", {})["api_key"] = os.environ["HLOKK_LLM_KEY"]
    if os.environ.get("HLOKK_LLM_URL"):
        config.setdefault("llm", {})["base_url"] = os.environ["HLOKK_LLM_URL"]
    if os.environ.get("HLOKK_LLM_MODEL"):
        config.setdefault("llm", {})["model"] = os.environ["HLOKK_LLM_MODEL"]

    # apply session override
    if args.focus:
        profile["session_focus"] = args.focus

    embed_config = config["embedding"]
    llm_config = config["llm"]
    rag_config = config.get("rag", {})

    # ------ resolve analysis mode ------
    mode = getattr(args, "mode", None)
    if not mode:
        # infer from profile depth setting
        depth = profile.get("output_preference", {}).get("depth", "standard")
        mode = "detailed" if depth == "deep_dive" else "overview"
    dual_call = (mode == "detailed")
    mode_desc = f"{mode} ({'dual LLM calls' if dual_call else 'single LLM call'})"
    console.print(f"  Analysis mode: [bold]{mode_desc}[/bold]")

    # ------ step 1: parse & classify PDFs ------
    total_steps = 8
    console.print(Panel(f"[bold]Step 1/{total_steps}[/bold] Parsing and classifying PDFs...", style="blue"))
    pdf_paths = [str(Path(p).resolve()) for p in args.pdfs]
    documents = classify_documents(pdf_paths, main_hint=args.main_hint)

    main_doc = documents["main"]
    supp_docs = documents["supplements"]

    if main_doc is None:
        console.print("[red]ERROR:[/red] Could not identify a main paper.")
        sys.exit(1)

    console.print(f"  Main paper: {main_doc.file_name} ({main_doc.page_count} pages)")
    for s in supp_docs:
        console.print(f"  Supplement: {s.file_name} ({s.page_count} pages, {s.sub_type})")

    # extract sections from main paper (three-layer: font → regex → LLM fallback)
    main_sections = extract_sections_enhanced(
        main_doc.full_text,
        pdf_path=main_doc.file_path,
        llm_config=llm_config,
        doc=main_doc,
    )
    console.print(f"  Detected sections: {list(main_sections.keys())}")

    # extract all figure/table IDs mentioned in the text
    figure_table_ids = extract_figure_table_ids(main_doc.full_text)
    console.print(f"  Detected figure/table IDs: {len(figure_table_ids)} ({', '.join(figure_table_ids[:8])}{'...' if len(figure_table_ids) > 8 else ''})")

    # ------ step 2: chunk supplementary materials ------
    paper_id = paper_id_from_paths(pdf_paths)
    all_supp_chunks = []
    queries = []  # initialize early to avoid scope issues

    if supp_docs and not args.skip_rag:
        console.print(Panel(f"[bold]Step 2/{total_steps}[/bold] Chunking supplementary materials...", style="blue"))
        chunk_size = rag_config.get("chunk_size", 512)
        chunk_overlap = rag_config.get("chunk_overlap", 64)

        for sdoc in supp_docs:
            supp_sections = extract_sections_enhanced(
                sdoc.full_text, pdf_path=sdoc.file_path, doc=sdoc,
            )
            if len(supp_sections) <= 1:
                chunks = chunk_text(
                    sdoc.full_text, sdoc.file_name,
                    section=sdoc.sub_type,
                    max_tokens=chunk_size, overlap_tokens=chunk_overlap,
                )
            else:
                chunks = chunk_document_sections(
                    supp_sections, sdoc.file_name,
                    max_tokens=chunk_size, overlap_tokens=chunk_overlap,
                )
            all_supp_chunks.extend(chunks)

        console.print(f"  Total chunks: {len(all_supp_chunks)}")
    else:
        console.print(Panel(f"[bold]Step 2/{total_steps}[/bold] Skipping RAG (no supplements or --skip-rag).", style="yellow"))

    # ------ step 3: embed & store ------
    retrieved_chunks = []
    if all_supp_chunks:
        console.print(Panel(f"[bold]Step 3/{total_steps}[/bold] Embedding and storing in vector DB...", style="blue"))
        db_path = str(workspace / "data" / "chroma_db")
        db_client = init_store(db_path)

        col_prefix = rag_config.get("collection_prefix", "hlokk")
        col_name = store_chunks(
            all_supp_chunks, embed_config, db_client,
            collection_prefix=col_prefix, paper_id=paper_id,
        )
        console.print(f"  Collection: {col_name} ({len(all_supp_chunks)} chunks stored)")

        # ------ step 4: retrieve relevant chunks ------
        console.print(Panel(f"[bold]Step 4/{total_steps}[/bold] Retrieving relevant supplement chunks...", style="blue"))
        queries = build_queries_from_profile(profile)

        if profile.get("session_focus"):
            queries.insert(0, profile["session_focus"])

        top_k = rag_config.get("top_k", 15)
        retrieved_chunks = retrieve_relevant(
            queries, embed_config, db_client, col_name, top_k=top_k,
        )
        console.print(f"  Retrieved {len(retrieved_chunks)} chunks from {len(queries)} queries")
    else:
        console.print(Panel(f"[bold]Step 3/{total_steps}[/bold] Skipping embedding (no chunks).", style="yellow"))
        console.print(Panel(f"[bold]Step 4/{total_steps}[/bold] Skipping retrieval (no chunks).", style="yellow"))

    # ------ step 5: LLM structured understanding ------
    lm_label = "Step 5a + 5b (Foundation + Deep)" if mode == "detailed" else "Step 5"
    console.print(Panel(
        f"[bold]{lm_label}[/bold] Calling LLM for structured paper understanding ({mode_desc})...",
        style="blue"
    ))
    understanding = understand_paper(
        main_text=main_doc.full_text,
        retrieved_chunks=retrieved_chunks,
        profile=profile,
        llm_config=llm_config,
        mode=mode,
        figure_table_ids=figure_table_ids,
        doc_type_override=getattr(args, "doc_type", None),
    )

    # Resolve doc_type for downstream use (gap detection, report rendering)
    doc_type = understanding.get("_doc_type", "article")
    console.print(f"  Document type: [bold]{doc_type}[/bold]")

    if understanding.get("_parse_error") or understanding.get("_foundation_failed"):
        console.print("[yellow]WARNING:[/yellow] LLM returned non-JSON response. Raw text preserved in output.")
    elif understanding.get("_deep_pass_error"):
        console.print(f"[yellow]WARNING:[/yellow] Deep analysis pass failed: {understanding['_deep_pass_error']}. Foundation results available.")
    elif understanding.get("_deep_pass"):
        console.print("  [green]✓[/green] Foundation pass + Deep analysis pass both completed.")
    else:
        console.print("  [green]✓[/green] Foundation pass completed.")

    # ------ enrich with paper URLs for clickable HTML header ------
    understanding = _enrich_paper_urls(
        understanding,
        zotero_key=getattr(args, "zotero_key", None),
        paper_url_override=getattr(args, "paper_url", None),
    )
    if understanding.get("paper_url"):
        console.print(f"  Paper URL: [dim]{understanding['paper_url']}[/dim]")

    # ------ step 6: extract figure page images (default disabled) ------
    figure_data = None
    if args.with_figures:
        fig_args = understanding.get("figure_arguments", [])
        if fig_args:
            console.print(Panel(f"[bold]Step 6/{total_steps}[/bold] Extracting figure page images...", style="blue"))
            figure_data = extract_figures_for_report(
                main_doc.file_path, fig_args, dpi=150,
            )
            n_pages = len(figure_data.get("page_images", {}))
            total_kb = figure_data.get("total_size_kb", 0)
            console.print(f"  Rendered {n_pages} pages ({total_kb:.0f} KB total)")
        else:
            console.print(Panel(f"[bold]Step 6/{total_steps}[/bold] No figure_arguments from LLM, skipping.", style="yellow"))
    else:
        console.print(Panel(f"[bold]Step 6/{total_steps}[/bold] Figure rendering disabled by default. Use --with-figures to enable.", style="yellow"))

    # ------ step 7: coverage gap detection ------
    console.print(Panel(f"[bold]Step 7/{total_steps}[/bold] Running coverage gap detection...", style="blue"))
    coverage_gaps = detect_coverage_gaps(
        understanding=understanding,
        detected_sections=list(main_sections.keys()),
        detected_figure_ids=figure_table_ids,
        rag_stats={
            "total_chunks": len(all_supp_chunks),
            "retrieved_chunks": len(retrieved_chunks),
        },
        doc_type=doc_type,
    )
    score = coverage_gaps["overall_score"]
    score_color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
    console.print(f"  Coverage score: [{score_color}]{score:.0%}[/{score_color}]")
    if coverage_gaps["figures_coverage"]["missing_main"]:
        console.print(f"  [yellow]Missing main figures:[/yellow] {', '.join(coverage_gaps['figures_coverage']['missing_main'])}")
    if coverage_gaps["sections_coverage"]["missing"]:
        console.print(f"  [yellow]Missing sections:[/yellow] {', '.join(coverage_gaps['sections_coverage']['missing'])}")

    # ------ assemble output ------
    timestamp = get_timestamp()
    output = {
        "hlokk_version": "2.2.0",
        "generated_at": timestamp,
        "paper_id": paper_id,
        "analysis_mode": mode,
        "doc_type": doc_type,
        "dual_call": understanding.get("_deep_pass", False),
        "zotero_key": getattr(args, "zotero_key", None),
        "input_files": {
            "main": main_doc.to_dict(),
            "supplements": [s.to_dict() for s in supp_docs],
        },
        "main_sections": list(main_sections.keys()),
        "detected_figure_table_ids": figure_table_ids,
        "rag_stats": {
            "total_chunks": len(all_supp_chunks),
            "retrieved_chunks": len(retrieved_chunks),
            "queries_used": len(queries),
        },
        "coverage_gaps": coverage_gaps,
        "understanding": understanding,
        "user_profile_snapshot": {
            "research_field": profile.get("research_field"),
            "pain_points": profile.get("pain_points", []),
            "focus": profile.get("output_preference", {}).get("focus"),
            "session_focus": profile.get("session_focus"),
        },
    }

    # save JSON
    output_dir = ensure_dir(str(workspace / "outputs"))
    title_slug = understanding.get("title", "paper")[:60].replace(" ", "_").replace("/", "_")
    base_name = f"hlokk_{title_slug}_{timestamp}"
    json_file = str(output_dir / f"{base_name}.json")
    save_json(output, json_file)
    console.print(f"  JSON: {json_file}")

    # ------ step 8: generate interactive HTML report ------
    console.print(Panel(f"[bold]Step 8/{total_steps}[/bold] Generating interactive HTML report...", style="blue"))
    html_file = str(output_dir / f"{base_name}.html")
    generate_html_report(output, figure_data=figure_data, output_path=html_file)
    console.print(f"  HTML: {html_file}")

    console.print(f"\n[green]Done![/green] Reports saved to: {output_dir}/")

    # ------ optional step: ingest into HlokkObsidian ------
    if getattr(args, "ingest", False):
        console.print(Panel("[bold]Ingest Step[/bold] Converting to Obsidian entity page...", style="cyan"))
        ingest_script = Path(__file__).parent / "hlokk_to_wiki.py"
        if not ingest_script.exists():
            console.print("[red]ERROR:[/red] hlokk_to_wiki.py not found — skipping ingest.")
        else:
            import subprocess
            python = sys.executable
            ingest_cmd = [python, str(ingest_script), "--json", json_file]
            if getattr(args, "zotero_key", None):
                ingest_cmd += ["--zotero-key", args.zotero_key]
            if getattr(args, "theme", None):
                ingest_cmd += ["--theme", args.theme]
            vault_path = config.get("vault_path") or profile.get("vault_path")
            if vault_path:
                ingest_cmd += ["--vault", vault_path]
            result = subprocess.run(ingest_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                console.print(f"[green]✓[/green] Obsidian page created.\n{result.stdout.strip()}")
            else:
                console.print(f"[red]✗[/red] Ingest failed:\n{result.stderr.strip()}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Hlokk - Academic paper interpretation pipeline"
    )
    parser.add_argument(
        "--pdfs", nargs="+", required=True,
        help="PDF file paths (main paper + supplements)"
    )
    parser.add_argument(
        "--workspace", default=".",
        help="Workspace directory containing config.yaml and profile.yaml"
    )
    parser.add_argument(
        "--focus", default=None,
        help="Session focus override (e.g., 'spatial deconvolution methods')"
    )
    parser.add_argument(
        "--main-hint", default=None,
        help="Filename substring to force-identify the main paper"
    )
    parser.add_argument(
        "--skip-rag", action="store_true",
        help="Skip RAG pipeline (only use main text)"
    )
    parser.add_argument(
        "--with-figures", action="store_true",
        help="Enable figure page rendering in HTML report (slower, larger output). Default: disabled."
    )
    parser.add_argument(
        "--mode", choices=["overview", "detailed"], default=None,
        help="Analysis depth: overview (fast, single LLM call) or detailed (dual LLM calls, full coverage). "
             "If not specified, inferred from profile.yaml depth setting."
    )
    parser.add_argument(
        "--doc-type", choices=["article", "review", "benchmark", "commentary", "atlas"],
        default=None, dest="doc_type",
        help="Override document type detection. If not specified, auto-detected from "
             "Foundation Pass paper_contribution_type. Use for reviews, benchmarks, "
             "commentaries, or atlas/dataset papers when auto-detection might fail."
    )
    parser.add_argument(
        "--zotero-key", default=None, dest="zotero_key",
        help="Zotero item key for linking to knowledge base (e.g., ABC12345)"
    )
    parser.add_argument(
        "--ingest", action="store_true",
        help="After pipeline, convert output to Obsidian entity page via hlokk_to_wiki.py"
    )
    parser.add_argument(
        "--theme", default=None,
        help="HlokkObsidian theme path for ingest (e.g., 'Genomics/Data-Analysis'). "
             "If omitted, page is placed in 'Reading-Queue/Uncategorized/'"
    )
    parser.add_argument(
        "--paper-url", default=None, dest="paper_url",
        help="Explicit article URL for the HTML header (e.g. from Zotero MCP). "
             "Overrides DOI-based resolution."
    )
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
