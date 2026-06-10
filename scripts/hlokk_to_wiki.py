#!/usr/bin/env python3
"""
hlokk_to_wiki.py — Hlokk JSON → HlokkObsidian v4.1 Bridge (Multi-Type)

Converts the structured JSON output from hlokk-skill into:
  1. Source card (thin) → Sources/{short-name}.md
  2. Raw staging file (full analysis) → raw/{ZOTERO_KEY}_{short-name}.md (status: raw)

The raw staging file preserves the complete deep analysis. Phase B of hlokk-wiki
will later pick it up and extract 3-5 atomic knowledge notes into Knowledge/{theme}/.

v4.1: Multi-document-type support (article, review, benchmark, atlas, commentary).
      Type-aware source cards, narrative renderers, figure arguments, methods, and recommendations.

Usage:
    python hlokk_to_wiki.py --json /path/to/hlokk_output.json
    python hlokk_to_wiki.py --json output.json --vault /path/to/HlokkObsidian
    python hlokk_to_wiki.py --json output.json --zotero-key ABC123 --theme "Genomics/Data-Analysis"
    python hlokk_to_wiki.py --json output.json --dry-run

This script is called automatically when --ingest is passed to hlokk_main.py.
"""
import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path


# ============================================================
# CONTRIBUTION TYPE → BADGE LABEL
# ============================================================
CONTRIBUTION_BADGES = {
    "new_mechanism":          "🔬 新机制",
    "new_method":             "⚙️ 新方法",
    "new_framework":          "🏗️ 新框架",
    "new_dataset":            "📦 新数据集",
    "atlas":                  "🗺️ 图谱",
    "resource":               "📦 资源",
    "incremental_validation": "✅ 增量验证",
    "review_synthesis":       "📖 综述",
    "benchmark":              "📊 Benchmark",
    "commentary":             "💬 评论",
    "editorial":              "📝 社论",
    "perspective":            "🔭 观点",
}

# ============================================================
# DOC TYPE CONFIG — type-aware labels and card types
# ============================================================
DOC_TYPE_CONFIG = {
    "article": {
        "label": "原创研究",
        "card_type": "source",
        "knowledge_card_label": "知识点索引卡",
    },
    "review": {
        "label": "综述",
        "card_type": "knowledge-map-index",
        "knowledge_card_label": "知识图谱索引卡",
    },
    "benchmark": {
        "label": "方法评测",
        "card_type": "method-selection",
        "knowledge_card_label": "方法选型卡",
    },
    "atlas": {
        "label": "数据资源",
        "card_type": "resource-evaluation",
        "knowledge_card_label": "资源评估卡",
    },
    "commentary": {
        "label": "评论观点",
        "card_type": "argument-card",
        "knowledge_card_label": "论证分析卡",
    },
}

# Figure argument role labels (extended for all types)
ARGUMENT_ROLE_LABELS = {
    # article
    "phenomenon_establishment": "建立现象",
    "mechanism_proposal":       "提出机制",
    "mechanism_validation":     "验证机制",
    "clinical_relevance":       "临床意义",
    "methodological_demonstration": "方法展示",
    "negative_control":         "阴性对照",
    # review
    "conceptual_overview":      "概念总览",
    "data_summary":             "数据汇总",
    "comparison_table":         "对比表",
    "workflow_summary":         "流程总结",
    # benchmark
    "performance_comparison":   "性能对比",
    "scalability_test":         "可扩展性测试",
    "robustness_evaluation":    "鲁棒性评估",
    # atlas
    "quality_demonstration":    "质量展示",
    "biological_finding":       "生物学发现",
    "coverage_map":             "覆盖图谱",
    # commentary
    "rhetorical_illustration":  "修辞图示",
    "evidence_visualization":   "证据可视化",
}

# Default placement when no theme is specified
DEFAULT_THEME = "Computational-Methods/Foundation-Models"


# ============================================================
# DOC TYPE RESOLVER
# ============================================================

def _resolve_doc_type(output_data: dict) -> str:
    """Resolve document type from output metadata or Foundation Pass."""
    # Priority: top-level doc_type > _doc_type > infer from contribution_type
    dt = output_data.get("doc_type")
    if dt and dt in DOC_TYPE_CONFIG:
        return dt
    dt = output_data.get("understanding", {}).get("_doc_type")
    if dt and dt in DOC_TYPE_CONFIG:
        return dt
    # Fallback: infer from paper_contribution_type
    ct = output_data.get("understanding", {}).get("paper_contribution_type", "")
    if ct == "review_synthesis":
        return "review"
    elif ct == "benchmark":
        return "benchmark"
    elif ct in ("commentary", "editorial", "perspective"):
        return "commentary"
    elif ct in ("new_dataset", "atlas", "resource"):
        return "atlas"
    return "article"


# ============================================================
# SHORT NAME GENERATION
# ============================================================

def _make_short_name(title: str) -> str:
    """
    Generate a compact kebab-case filename from the paper title.
    Strategy:
      1. If title contains a colon, use the part before it (often the tool/method name).
      2. If that prefix is CamelCase or ALL-CAPS, use it directly.
      3. Otherwise, extract 2-5 meaningful words, drop stop words.
    """
    if not title:
        return "untitled"

    stop_words = {
        "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
        "with", "from", "by", "as", "is", "are", "was", "be", "via",
        "using", "based", "across", "into", "through", "within", "between",
    }

    # Strategy 1: colon-prefix (e.g., "scGPT: Toward ..." → "scGPT")
    if ":" in title:
        prefix = title.split(":")[0].strip()
        # If prefix looks like a tool name (short, CamelCase or ALLCAPS)
        if len(prefix.split()) <= 3 and re.search(r"[A-Z]{2}|[a-z][A-Z]", prefix):
            return re.sub(r"[^a-zA-Z0-9]+", "-", prefix).strip("-").lower()

    # Strategy 2: extract meaningful words
    words = re.findall(r"[A-Za-z0-9]+", title)
    meaningful = [w for w in words if w.lower() not in stop_words and len(w) > 1]
    selected = meaningful[:5]
    return "-".join(w.lower() for w in selected) if selected else "paper"


# ============================================================
# SOURCE CARD BUILDER (v4.1 — type-aware thin card for Sources/)
# ============================================================

def _build_type_summary(understanding: dict, doc_type: str) -> str:
    """Build a one-sentence type-aware summary for the source card."""
    fp = understanding.get("field_positioning", "")
    rc = understanding.get("reading_cipher", {})
    wtd = rc.get("WTD", "")

    if doc_type == "review":
        # For reviews, prefer research_question or review_scope
        rq = understanding.get("research_question", "")
        scope = understanding.get("review_scope", {})
        org = scope.get("organization_principle", "")
        n_papers = scope.get("paper_count", "")
        if rq:
            return rq
        if fp:
            suffix = f"（涵盖{n_papers}篇文献，{org}组织）" if n_papers else ""
            return f"{fp}{suffix}"
        return wtd or fp or "（综述待补充）"

    elif doc_type == "benchmark":
        bd = understanding.get("benchmark_design", {})
        problem = bd.get("problem_definition", "")
        n_methods = len(bd.get("methods_evaluated", []))
        if problem and n_methods:
            return f"{problem[:120]}（评测{n_methods}种方法）"
        return problem or fp or wtd or "（评测待补充）"

    elif doc_type == "atlas":
        rd = understanding.get("resource_design", {})
        rtype = rd.get("resource_type", "")
        organism = rd.get("organism", "")
        if rtype:
            return f"{rtype}资源（{organism}）" if organism else rtype
        return fp or wtd or "（资源待补充）"

    elif doc_type == "commentary":
        cs = understanding.get("commentary_structure", {})
        position = cs.get("main_position", "")
        trigger = cs.get("trigger", "")
        if position:
            return f"核心论点：{position}" + (f"（由{trigger[:40]}引发）" if trigger else "")
        return fp or wtd or "（评论待补充）"

    # Default: article
    return fp or wtd or "（待补充）"


def build_source_card(output_data: dict, zotero_key: str, theme_path: str) -> tuple[str, str]:
    """
    Build a thin source card for Sources/.
    Returns (page_content, suggested_filename).
    """
    understanding = output_data.get("understanding", {})
    concepts = understanding.get("concepts", [])

    title = understanding.get("title", "Untitled")
    short_name = _make_short_name(title)

    # Authors
    authors_raw = understanding.get("authors", "")
    authors_list = [a.strip() for a in re.split(r",|;", authors_raw) if a.strip()]

    # Year / journal
    journal_year = understanding.get("journal_year", "")
    year_match = re.search(r"\b(19|20)\d{2}\b", journal_year)
    year = year_match.group(0) if year_match else ""
    journal = re.sub(r",?\s*\b(19|20)\d{2}\b", "", journal_year).strip().rstrip(",").strip()

    doi = understanding.get("doi", "")
    contribution_type = understanding.get("paper_contribution_type", "")
    doc_type = _resolve_doc_type(output_data)
    dt_config = DOC_TYPE_CONFIG.get(doc_type, DOC_TYPE_CONFIG["article"])
    today = date.today().isoformat()

    # Tags from concepts
    tags_str = ", ".join(concepts[:8]) if concepts else ""

    # Escape quotes
    title_yaml = title.replace('"', '\\"')
    journal_yaml = journal.replace('"', '\\"') if journal else ""

    # --- Build frontmatter ---
    fm_lines = [
        "---",
        f'title: "{title_yaml}"',
        f'type: {dt_config["card_type"]}',
        f'aliases: ["{short_name}"]',
    ]
    if authors_list:
        fm_lines.append(f'authors: [{", ".join(authors_list[:5])}]')
    if year:
        fm_lines.append(f"year: {year}")
    if journal:
        fm_lines.append(f'journal: "{journal_yaml}"')
    if doi:
        fm_lines.append(f'doi: "{doi}"')
    if zotero_key:
        fm_lines.append(f"zotero_key: {zotero_key}")
        fm_lines.append(f'zotero_link: "zotero://select/items/{zotero_key}"')
    if tags_str:
        fm_lines.append(f"tags: [{tags_str}]")
    if theme_path:
        fm_lines.append(f'themes: ["{theme_path}"]')
    if contribution_type:
        fm_lines.append(f"paper_contribution_type: {contribution_type}")
    if doc_type != "article":
        fm_lines.append(f"doc_type: {doc_type}")
    fm_lines.append(f"date_processed: {today}")
    fm_lines.append("---")

    frontmatter = "\n".join(fm_lines)

    # --- Build body (type-aware summary + knowledge card placeholder) ---
    summary = _build_type_summary(understanding, doc_type)
    badge = CONTRIBUTION_BADGES.get(contribution_type, "")
    badge_prefix = f"{badge} " if badge else ""
    card_label = dt_config["knowledge_card_label"]

    body_lines = [
        f"> {badge_prefix}{summary}",
        "",
    ]

    # Type-specific placeholder section
    if doc_type == "review":
        body_lines += [
            f"## {card_label}",
            "",
            "（待 Phase B 知识点提取后填充：主题线索 [[wikilink]]、方法族 [[wikilink]]、共识与争议 [[wikilink]]）",
            "",
        ]
    elif doc_type == "benchmark":
        body_lines += [
            f"## {card_label}",
            "",
            "（待 Phase B 知识点提取后填充：被评测方法 [[wikilink]]、评测结论 [[wikilink]]、方法推荐 [[wikilink]]）",
            "",
        ]
    elif doc_type == "atlas":
        body_lines += [
            f"## {card_label}",
            "",
            "（待 Phase B 知识点提取后填充：资源设计 [[wikilink]]、数据质量 [[wikilink]]、复用指南 [[wikilink]]）",
            "",
        ]
    elif doc_type == "commentary":
        body_lines += [
            f"## {card_label}",
            "",
            "（待 Phase B 知识点提取后填充：核心论点 [[wikilink]]、论证链 [[wikilink]]、领域影响 [[wikilink]]）",
            "",
        ]
    else:
        body_lines += [
            "## 衍生知识点",
            "",
            "（待 Phase B 知识点提取后自动填充 [[wikilink]]）",
            "",
        ]

    content = frontmatter + "\n\n" + "\n".join(body_lines)
    filename = short_name + ".md"

    return content, filename


# ============================================================
# RAW STAGING — UNIVERSAL RENDERERS
# ============================================================

def _render_core_problem(understanding: dict, doc_type: str) -> str:
    """Render core problem section (type-aware labels)."""
    rc = understanding.get("reading_cipher", {})
    wtd = rc.get("WTD", "")
    gap = rc.get("GAP", "")
    rat = rc.get("RAT", "")
    fp = understanding.get("field_positioning", "")
    contrib_type = understanding.get("paper_contribution_type", "")
    badge = CONTRIBUTION_BADGES.get(contrib_type, "")
    dt_config = DOC_TYPE_CONFIG.get(doc_type, DOC_TYPE_CONFIG["article"])

    # Section header varies by type
    header_map = {
        "article": "核心问题与贡献",
        "review": "综述目标与范围",
        "benchmark": "评测目标与设计",
        "atlas": "资源目标与设计",
        "commentary": "核心论点与立场",
    }
    section_header = header_map.get(doc_type, "核心问题与贡献")

    parts = [f"## {section_header}\n"]
    if badge:
        parts.append(f"> [!info] {badge} · {dt_config['label']}\n")

    if doc_type == "review":
        rq = understanding.get("research_question", "")
        scope = understanding.get("review_scope", {})
        if rq:
            parts.append(f"**综述问题**：{rq}\n")
        if scope:
            inc = scope.get("inclusion_criteria", "")
            span = scope.get("time_span", "")
            n = scope.get("paper_count", "")
            org = scope.get("organization_principle", "")
            if inc: parts.append(f"\n**纳入标准**：{inc}\n")
            if span: parts.append(f"\n**时间跨度**：{span}\n")
            if n: parts.append(f"\n**文献数量**：{n} 篇\n")
            if org: parts.append(f"\n**组织方式**：{org}\n")
    elif doc_type == "benchmark":
        bd = understanding.get("benchmark_design", {})
        pd_text = bd.get("problem_definition", "")
        if pd_text:
            parts.append(f"**评测问题**：{pd_text}\n")
        protocol = bd.get("evaluation_protocol", "")
        gt = bd.get("ground_truth", "")
        if protocol: parts.append(f"\n**评测协议**：{protocol}\n")
        if gt: parts.append(f"\n**Ground Truth**：{gt}\n")
    elif doc_type == "atlas":
        rd = understanding.get("resource_design", {})
        rtype = rd.get("resource_type", "")
        org = rd.get("organism", "")
        tissue = rd.get("tissue_or_system", "")
        if rtype: parts.append(f"**资源类型**：{rtype}\n")
        if org: parts.append(f"\n**物种**：{org}\n")
        if tissue: parts.append(f"\n**组织/系统**：{tissue}\n")
    elif doc_type == "commentary":
        cs = understanding.get("commentary_structure", {})
        trigger = cs.get("trigger", "")
        position = cs.get("main_position", "")
        cta = cs.get("call_to_action", "")
        if trigger: parts.append(f"**触发事件**：{trigger}\n")
        if position: parts.append(f"\n**核心立场**：{position}\n")
        if cta: parts.append(f"\n**呼吁行动**：{cta}\n")
    else:
        if wtd: parts.append(f"**研究问题**：{wtd}\n")
        if gap: parts.append(f"\n**研究空白**：{gap}\n")
        if rat: parts.append(f"\n**研究动机**：{rat}\n")

    if fp:
        parts.append(f"\n**领域定位**：{fp}\n")
    return "\n".join(parts)


def _render_key_findings(understanding: dict) -> str:
    """Render key findings (universal)."""
    findings = understanding.get("key_findings", [])
    if not findings:
        return ""

    lines = ["## 核心发现\n"]
    for i, f in enumerate(findings, 1):
        text = f.get("finding", "")
        strength = f.get("strength", "")
        etype = f.get("evidence_type", "")
        loc = f.get("evidence_location", "")

        strength_tag = {"strong": "🟢", "moderate": "🟡", "weak": "🔴"}.get(strength, "")
        infer_tag = " *(推断)*" if etype == "inferred" else ""
        loc_tag = f" `{loc}`" if loc else ""

        lines.append(f"{i}. {strength_tag} {text}{infer_tag}{loc_tag}")

    return "\n".join(lines) + "\n"


def _render_reading_cipher(understanding: dict) -> str:
    """Render reading cipher (universal)."""
    rc = understanding.get("reading_cipher", {})
    if not rc:
        return ""

    def _fmt(val):
        if isinstance(val, list):
            return "；".join(val) if val else "-"
        return val or "-"

    cipher_order = [
        ("WTD",  "他们要做什么"),
        ("SPL",  "前人工作综述"),
        ("CPL",  "前人工作批评"),
        ("GAP",  "研究空白"),
        ("RAT",  "研究动机"),
        ("ROF",  "研究结果"),
        ("RCL",  "与文献一致"),
        ("RTC",  "与文献相悖"),
        ("WTDD", "他们做了什么"),
        ("RFW",  "未来方向"),
    ]

    lines = [
        "## 阅读密码速览\n",
        "| 密码 | 含义 | 内容 |",
        "|------|------|------|",
    ]
    for code, label in cipher_order:
        val = _fmt(rc.get(code, "-"))
        if len(val) > 120:
            val = val[:117] + "…"
        val = val.replace("|", "｜")
        lines.append(f"| **{code}** | {label} | {val} |")

    return "\n".join(lines) + "\n"


def _render_limitations(understanding: dict, doc_type: str) -> str:
    """Render limitations (type-aware: picks up type-specific limitation fields)."""
    limitations = understanding.get("limitations", [])
    lob = understanding.get("limitations_of_benchmark", [])
    lag = understanding.get("limitations_and_gaps", [])
    rfw = understanding.get("reading_cipher", {}).get("RFW", [])

    has_any = limitations or lob or lag or rfw
    if not has_any:
        return ""

    lines = ["## 局限与开放问题\n"]

    if limitations:
        lines.append("**已识别局限：**\n")
        for lim in limitations:
            text = lim.get("limitation", "")
            etype = lim.get("evidence_type", "")
            loc = lim.get("evidence_location", "")
            tag = " *(文章承认)*" if etype == "stated" else " *(分析判断)*"
            loc_tag = f" `{loc}`" if loc else ""
            lines.append(f"- {text}{tag}{loc_tag}")
        lines.append("")

    if lob:
        lines.append("**评测设计局限：**\n")
        for lim in lob:
            text = lim.get("limitation", "")
            impact = lim.get("impact", "")
            etype = lim.get("evidence_type", "")
            tag = " *(文章承认)*" if etype == "stated" else ""
            impact_tag = f" — *影响: {impact}*" if impact else ""
            lines.append(f"- {text}{tag}{impact_tag}")
        lines.append("")

    if lag:
        lines.append("**资源局限：**\n")
        for lim in lag:
            text = lim.get("limitation", "")
            cat = lim.get("category", "")
            impact = lim.get("impact", "")
            cat_tag = f" *[{cat}]*" if cat else ""
            impact_tag = f" — *影响: {impact}*" if impact else ""
            lines.append(f"- {text}{cat_tag}{impact_tag}")
        lines.append("")

    if rfw:
        lines.append("**未来方向（作者建议）：**\n")
        if isinstance(rfw, list):
            for r in rfw:
                lines.append(f"- {r}")
        else:
            lines.append(f"- {rfw}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# RAW STAGING — TYPE-SPECIFIC RENDERERS
# ============================================================

# --- ARTICLE: narrative map + methods + figures ---

def _render_narrative_map_article(understanding: dict) -> str:
    """Render research_threads as a narrative map (article)."""
    threads = understanding.get("research_threads", [])
    if not threads:
        return ""

    lines = ["## 论文叙事脉络\n"]
    sorted_threads = sorted(threads, key=lambda t: t.get("narrative_stage", 99))

    for thread in sorted_threads:
        stage = thread.get("narrative_stage", "")
        label = thread.get("stage_label", "")
        main_q = thread.get("main_question", "")

        stage_header = f"**Stage {stage}** — {label}" if stage and label else (label or main_q)
        lines.append(f"### {stage_header}\n")
        if main_q and main_q != stage_header:
            lines.append(f"**核心问题**：{main_q}\n")

        sub_questions = thread.get("sub_questions", [])
        for sq in sub_questions:
            sub_q = sq.get("sub_question", "")
            if sub_q:
                lines.append(f"\n- **{sub_q}**")
            for ep in sq.get("evidence_points", []):
                claim = ep.get("claim", "")
                loc = ep.get("evidence_location", "")
                strength = ep.get("strength", "")
                etype = ep.get("evidence_type", "")
                figs = ", ".join(ep.get("supporting_figures", []))

                tag = f"[{strength}]" if strength else ""
                infer_tag = " *(推断)*" if etype == "inferred" else ""
                loc_tag = f" `{loc}`" if loc else ""
                fig_tag = f" — {figs}" if figs else ""

                lines.append(f"  - {claim}{infer_tag} {tag}{loc_tag}{fig_tag}")
        lines.append("")

    return "\n".join(lines)


def _render_methods_article(understanding: dict) -> str:
    """Render core_methods table (article)."""
    methods = understanding.get("core_methods", [])
    stats = understanding.get("statistical_methods", [])
    if not methods and not stats:
        return ""

    parts = []
    if methods:
        lines = [
            "## 方法速查\n",
            "| 工具/方法 | 版本 | 用途 | 关键参数 | 证据位置 | 对应结果 |",
            "|-----------|------|------|----------|----------|----------|",
        ]
        for m in methods:
            tool = m.get("tool") or m.get("name", "-")
            version = m.get("version") or "-"
            purpose = (m.get("purpose") or "-")[:60]
            params = str(m.get("parameters") or "-")[:40]
            loc = (m.get("evidence_location") or "-")[:30]
            results = "; ".join((m.get("supports_figures") or [])[:3]) or "-"
            lines.append(f"| {tool} | {version} | {purpose} | {params} | {loc} | {results} |")
        parts.append("\n".join(lines) + "\n")

    if stats:
        slines = [
            "### 统计方法\n",
            "| 检验 | 校正 | 阈值 | 场景 |",
            "|------|------|------|------|",
        ]
        for s in stats:
            slines.append(
                f"| {s.get('test','-')} | {s.get('correction','-')} "
                f"| {s.get('threshold','-')} | {s.get('context','-')} |"
            )
        parts.append("\n".join(slines) + "\n")

    return "\n".join(parts)


def _render_figure_arguments_article(understanding: dict) -> str:
    """Render figure arguments for article type."""
    figs = understanding.get("figure_arguments", [])
    if not figs:
        return ""

    lines = ["## 图表论证\n"]
    for fig in figs:
        fid = fig.get("figure_id", "Figure ?")
        role = fig.get("argument_role", "")
        role_label = ARGUMENT_ROLE_LABELS.get(role, role)
        sub_arg = fig.get("sub_argument", "")
        necessity = fig.get("logical_necessity", "")
        weakness = fig.get("weakness", "")
        weakness_type = fig.get("weakness_type", "")
        loc = fig.get("evidence_location", "")

        role_badge = f"**[{role_label}]**" if role_label else ""
        loc_tag = f" `{loc}`" if loc else ""

        lines.append(f"### {fid} {role_badge}{loc_tag}\n")
        if sub_arg:
            lines.append(f"**分论点**：{sub_arg}\n")
        if necessity:
            lines.append(f"**论证必要性**：{necessity}\n")
        if weakness:
            w_tag = " *(文章承认)*" if weakness_type == "stated_limitation" else " *(分析判断)*"
            lines.append(f"\n> [!warning] 证据局限{w_tag}\n> {weakness}\n")
        lines.append("")

    return "\n".join(lines)


# --- REVIEW: thematic threads + consensus/debate + methods landscape ---

def _render_narrative_review(understanding: dict) -> str:
    """Render thematic_threads + consensus_and_debate (review)."""
    threads = understanding.get("thematic_threads", [])
    cd = understanding.get("consensus_and_debate", {})

    if not threads and not cd:
        return ""

    parts = []

    if threads:
        lines = ["## 主题线索\n"]
        for t in threads:
            order = t.get("theme_order", "")
            name = t.get("theme_name", "")
            scope = t.get("theme_scope", "")
            evo = t.get("evolution_narrative", "")
            state = t.get("current_state", "")
            oq = t.get("open_questions", "")
            refs = t.get("key_references", [])

            lines.append(f"### #{order} {name}\n")
            if scope: lines.append(f"**范围**：{scope}\n")
            if evo: lines.append(f"**演进**：{evo}\n")
            if state: lines.append(f"**当前水平**：{state}\n")
            if oq:
                if isinstance(oq, list):
                    lines.append(f"**开放问题**：{'; '.join(oq)}\n")
                else:
                    lines.append(f"**开放问题**：{oq}\n")
            if refs:
                lines.append(f"*关键引用：{'; '.join(refs[:5])}*\n")
        parts.append("\n".join(lines))

    if cd:
        lines = ["## 领域共识与争议\n"]
        for c in cd.get("consensus_points", []):
            pt = c.get("point", "")
            es = c.get("evidence_strength", "")
            badge = {"strong": "🟢", "moderate": "🟡"}.get(es, "")
            lines.append(f"- {badge} {pt}\n")
        for d in cd.get("debate_points", []):
            pt = d.get("point", "")
            a = d.get("camp_a", "")
            b = d.get("camp_b", "")
            status = d.get("resolution_status", "")
            lines.append(f"\n**争议**：{pt}\n")
            lines.append(f"- **A方**：{a}")
            lines.append(f"- **B方**：{b}")
            lines.append(f"- *状态：{status}*\n")
        parts.append("\n".join(lines))

    return "\n".join(parts)


def _render_methods_review(understanding: dict) -> str:
    """Render methods_landscape (review)."""
    ml = understanding.get("methods_landscape", [])
    if not ml:
        return ""

    lines = ["## 方法族图谱\n"]
    for m in ml:
        family = m.get("method_family", "")
        tools = m.get("representative_tools", [])
        strengths = m.get("strengths", "")
        limits = m.get("limitations", "")
        best = m.get("best_for", "")
        loc = m.get("evidence_location", "")

        lines.append(f"### {family}\n")
        if tools: lines.append(f"**代表工具**：{', '.join(tools)}\n")
        if strengths: lines.append(f"**优势**：{strengths}\n")
        if limits: lines.append(f"**局限**：{limits}\n")
        if best: lines.append(f"**最适场景**：{best}\n")
        if loc: lines.append(f"*{loc}*\n")
    return "\n".join(lines)


def _render_figure_arguments_review(understanding: dict) -> str:
    """Render figure arguments for review type."""
    figs = understanding.get("figure_arguments", [])
    if not figs:
        return ""

    lines = ["## 图表组织\n"]
    for fig in figs:
        fid = fig.get("figure_id", "Figure ?")
        role = fig.get("argument_role", "")
        role_label = ARGUMENT_ROLE_LABELS.get(role, role)
        desc = fig.get("content_description", "")
        org_func = fig.get("organizational_function", "")
        covers = fig.get("covers_themes", [])
        loc = fig.get("evidence_location", "")

        role_badge = f"**[{role_label}]**" if role_label else ""
        loc_tag = f" `{loc}`" if loc else ""

        lines.append(f"### {fid} {role_badge}{loc_tag}\n")
        if desc: lines.append(f"**内容**：{desc}\n")
        if org_func: lines.append(f"**组织功能**：{org_func}\n")
        if covers: lines.append(f"**关联主题**：{', '.join(covers)}\n")
    return "\n".join(lines)


# --- BENCHMARK: evaluation threads + rankings + benchmark_design ---

def _render_narrative_benchmark(understanding: dict) -> str:
    """Render evaluation_threads + rankings (benchmark)."""
    et = understanding.get("evaluation_threads", [])
    rr = understanding.get("rankings_and_recommendations", {})

    if not et and not rr:
        return ""

    parts = []

    if et:
        lines = ["## 评测维度分析\n"]
        for t in et:
            order = t.get("dimension_order", "")
            name = t.get("dimension_name", "")
            findings = t.get("main_findings", "")
            top = t.get("top_performers", [])
            bot = t.get("bottom_performers", [])
            surprise = t.get("surprising_results", "")
            confound = t.get("confounding_factors", "")

            lines.append(f"### 维度 {order}: {name}\n")
            if findings: lines.append(f"**主要发现**：{findings}\n")
            if top: lines.append(f"**最佳方法**：{', '.join(top)}\n")
            if bot: lines.append(f"**最差方法**：{', '.join(bot)}\n")
            if surprise: lines.append(f"**意外发现**：{surprise}\n")
            if confound: lines.append(f"**混杂因素**：{confound}\n")
        parts.append("\n".join(lines))

    if rr:
        lines = ["## 排名与推荐\n"]
        ranking = rr.get("overall_ranking", [])
        if ranking:
            lines += [
                "| 排名 | 方法 | 优势 | 劣势 | 最佳场景 |",
                "|------|------|------|------|----------|",
            ]
            for r in ranking:
                lines.append(
                    f"| #{r.get('rank','')} | {r.get('method','-')} "
                    f"| {r.get('strengths','-')} | {r.get('weaknesses','-')} "
                    f"| {r.get('best_scenario','-')} |"
                )
            lines.append("")
        guide = rr.get("practical_guidelines", "")
        if guide:
            lines.append(f"> [!tip] 实践指南\n> {guide}\n")
        parts.append("\n".join(lines))

    return "\n".join(parts)


def _render_methods_benchmark(understanding: dict) -> str:
    """Render benchmark_design tables + methodological_insights."""
    bd = understanding.get("benchmark_design", {})
    mi = understanding.get("methodological_insights", {})
    if not bd and not mi:
        return ""

    parts = []

    if bd:
        lines = ["## 评测设计\n"]

        methods = bd.get("methods_evaluated", [])
        if methods:
            lines += [
                "### 被评测方法\n",
                "| 方法 | 版本 | 类别 | 关键参数 |",
                "|------|------|------|----------|",
            ]
            for m in methods:
                lines.append(
                    f"| {m.get('name','-')} | {m.get('version','-')} "
                    f"| {m.get('category','-')} | {m.get('key_parameters','-')} |"
                )
            lines.append("")

        datasets = bd.get("datasets", [])
        if datasets:
            lines += [
                "### 评测数据集\n",
                "| 名称 | 来源 | 特征 | 选用理由 |",
                "|------|------|------|----------|",
            ]
            for d in datasets:
                lines.append(
                    f"| {d.get('name','-')} | {d.get('source','-')} "
                    f"| {d.get('characteristics','-')} | {d.get('why_chosen','-')} |"
                )
            lines.append("")

        metrics = bd.get("metrics", [])
        if metrics:
            lines += [
                "### 评价指标\n",
                "| 指标 | 定义 | 衡量什么 |",
                "|------|------|----------|",
            ]
            for m in metrics:
                lines.append(
                    f"| {m.get('name','-')} | {m.get('definition','-')} "
                    f"| {m.get('measures_what','-')} |"
                )
            lines.append("")

        parts.append("\n".join(lines))

    if mi:
        lines = ["## 方法学洞察\n"]
        if mi.get("what_matters"): lines.append(f"**关键因素**：{mi['what_matters']}\n")
        if mi.get("what_doesnt_matter"): lines.append(f"**非关键因素**：{mi['what_doesnt_matter']}\n")
        if mi.get("parameter_sensitivity"): lines.append(f"**参数敏感性**：{mi['parameter_sensitivity']}\n")
        if mi.get("computational_considerations"): lines.append(f"**计算考量**：{mi['computational_considerations']}\n")
        parts.append("\n".join(lines))

    return "\n".join(parts)


def _render_figure_arguments_benchmark(understanding: dict) -> str:
    """Render figure arguments for benchmark type."""
    figs = understanding.get("figure_arguments", [])
    if not figs:
        return ""

    lines = ["## 评测图表\n"]
    for fig in figs:
        fid = fig.get("figure_id", "Figure ?")
        role = fig.get("argument_role", "")
        role_label = ARGUMENT_ROLE_LABELS.get(role, role)
        compared = fig.get("compared_methods", [])
        dim = fig.get("evaluation_dimension", "")
        takeaway = fig.get("key_takeaway", "")
        winner = fig.get("winner_context", "")
        desc = fig.get("content_description", "")
        loc = fig.get("evidence_location", "")

        role_badge = f"**[{role_label}]**" if role_label else ""
        loc_tag = f" `{loc}`" if loc else ""

        lines.append(f"### {fid} {role_badge}{loc_tag}\n")
        if desc: lines.append(f"**内容**：{desc}\n")
        if compared: lines.append(f"**对比方法**：{', '.join(compared)}\n")
        if dim: lines.append(f"**评测维度**：{dim}\n")
        if takeaway: lines.append(f"**核心结论**：{takeaway}\n")
        if winner: lines.append(f"**获胜条件**：{winner}\n")
    return "\n".join(lines)


# --- ATLAS: resource_design + annotation + accessibility ---

def _render_narrative_atlas(understanding: dict) -> str:
    """Render initial_biological_findings + resource_comparison (atlas)."""
    ibf = understanding.get("initial_biological_findings", [])
    rc = understanding.get("resource_comparison", {})

    if not ibf and not rc:
        return ""

    parts = []

    if ibf:
        lines = ["## 初步生物学发现\n"]
        for f in ibf:
            finding = f.get("finding", "")
            novelty = f.get("novelty", "")
            loc = f.get("evidence_location", "")
            methods = f.get("methods_used", "")
            badge = {"high": "🟢", "moderate": "🟡", "low": "🔴"}.get(novelty, "")
            loc_tag = f" `{loc}`" if loc else ""
            lines.append(f"- {badge} {finding}{loc_tag}")
            if methods: lines.append(f"  *方法: {methods}*")
        parts.append("\n".join(lines) + "\n")

    if rc:
        lines = ["## 资源对比\n"]
        compared = rc.get("compared_to", [])
        adv = rc.get("advantages", [])
        lim = rc.get("limitations", [])
        comp = rc.get("complementary_resources", [])
        if compared: lines.append(f"**对比对象**：{', '.join(compared)}\n")
        if adv: lines.append(f"**优势**：{'; '.join(adv)}\n")
        if lim: lines.append(f"**局限**：{'; '.join(lim)}\n")
        if comp: lines.append(f"**互补资源**：{', '.join(comp)}\n")
        parts.append("\n".join(lines))

    return "\n".join(parts)


def _render_methods_atlas(understanding: dict) -> str:
    """Render resource_design + annotation_assessment + data_accessibility."""
    rd = understanding.get("resource_design", {})
    aa = understanding.get("annotation_assessment", {})
    da = understanding.get("data_accessibility", {})
    if not rd and not aa and not da:
        return ""

    parts = []

    if rd:
        lines = ["## 资源设计\n"]
        tp = rd.get("technology_platform", {})
        if tp:
            lines.append(f"**技术平台**：{tp.get('sequencing_or_measurement', '')}")
            lines.append(f"**分辨率**：{tp.get('resolution', '')}")
            lines.append(f"**通量**：{tp.get('throughput', '')}\n")
        sd = rd.get("sample_design", {})
        if sd:
            lines.append(f"**样本数**：{sd.get('total_samples', '')}")
            lines.append(f"**选择标准**：{sd.get('selection_criteria', '')}")
            lines.append(f"**代表性**：{sd.get('representativeness', '')}")
            lines.append(f"**潜在偏倚**：{sd.get('potential_biases', '')}\n")
        pp = rd.get("preprocessing_pipeline", {})
        if pp:
            lines.append("### 预处理流程\n")
            lines.append(f"**比对**：{pp.get('alignment', '')}")
            lines.append(f"**质控**：{pp.get('qc_filters', '')}")
            lines.append(f"**标准化**：{pp.get('normalization', '')}")
            lines.append(f"**批次校正**：{pp.get('batch_correction', '')}\n")
        parts.append("\n".join(lines))

    if aa:
        lines = ["## 注释评估\n"]
        lines.append(f"**策略**：{aa.get('annotation_strategy', '')}")
        lines.append(f"**分类体系**：{aa.get('cell_type_taxonomy', '')}")
        lines.append(f"**深度**：{aa.get('annotation_depth', '')}")
        lines.append(f"**验证**：{aa.get('validation_method', '')}")
        lines.append(f"**置信度**：{aa.get('annotation_confidence', '')}")
        novel = aa.get("novel_types_discovered", [])
        if novel: lines.append(f"**新发现类型**：{', '.join(novel)}")
        parts.append("\n".join(lines) + "\n")

    if da:
        lines = ["## 数据可及性\n"]
        repos = da.get("repositories", [])
        acc = da.get("accession_numbers", [])
        fmt = da.get("data_formats", [])
        if repos: lines.append(f"**存储库**：{', '.join(repos)}")
        if acc: lines.append(f"**登录号**：{', '.join(acc)}")
        if fmt: lines.append(f"**格式**：{', '.join(fmt)}")
        lines.append(f"**元数据丰富度**：{da.get('metadata_richness', '')}")
        lines.append(f"**交互浏览器**：{da.get('interactive_browser', '无')}")
        lines.append(f"**许可**：{da.get('license', '未明确')}\n")
        parts.append("\n".join(lines))

    return "\n".join(parts)


def _render_figure_arguments_atlas(understanding: dict) -> str:
    """Render figure arguments for atlas type."""
    figs = understanding.get("figure_arguments", [])
    if not figs:
        return ""

    lines = ["## 资源图表\n"]
    for fig in figs:
        fid = fig.get("figure_id", "Figure ?")
        role = fig.get("argument_role", "")
        role_label = ARGUMENT_ROLE_LABELS.get(role, role)
        desc = fig.get("content_description", "")
        what = fig.get("demonstrates_what", "")
        subset = fig.get("data_subset", "")
        loc = fig.get("evidence_location", "")

        role_badge = f"**[{role_label}]**" if role_label else ""
        loc_tag = f" `{loc}`" if loc else ""

        lines.append(f"### {fid} {role_badge}{loc_tag}\n")
        if desc: lines.append(f"**内容**：{desc}\n")
        if what: lines.append(f"**证明了什么**：{what}\n")
        if subset: lines.append(f"**数据子集**：{subset}\n")
    return "\n".join(lines)


# --- COMMENTARY: argument chain + context ---

def _render_narrative_commentary(understanding: dict) -> str:
    """Render commentary_structure argument chain + context_and_stakes."""
    cs = understanding.get("commentary_structure", {})
    ctx = understanding.get("context_and_stakes", {})

    if not cs and not ctx:
        return ""

    parts = []

    if cs:
        lines = ["## 论证结构\n"]
        trigger = cs.get("trigger", "")
        position = cs.get("main_position", "")
        cta = cs.get("call_to_action", "")
        iic = cs.get("implications_if_correct", "")
        if trigger: lines.append(f"**触发事件**：{trigger}\n")
        if position: lines.append(f"**核心立场**：{position}\n")
        if cta: lines.append(f"**呼吁行动**：{cta}\n")
        if iic: lines.append(f"**若正确则**：{iic}\n")

        chain = cs.get("argument_chain", [])
        if chain:
            lines.append("### 论证链\n")
            for step in chain:
                s = step.get("step", "")
                claim = step.get("claim", "")
                ev = step.get("evidence", "")
                etype = step.get("evidence_type", "")
                lines.append(f"**步骤 {s}** [{etype}]：{claim}")
                lines.append(f"  *证据：{ev}*\n")
            parts.append("\n".join(lines))

        counter = cs.get("counterarguments_acknowledged", [])
        if counter:
            lines2 = ["### 承认的反对意见\n"]
            for ca in counter:
                lines2.append(f"- **{ca.get('counterargument', '')}**")
                lines2.append(f"  *回应：{ca.get('author_response', '')}*\n")
            parts.append("\n".join(lines2))

    if ctx:
        lines3 = ["## 背景与利害\n"]
        fc = ctx.get("field_context", "")
        ws = ctx.get("what_is_at_stake", "")
        tl = ctx.get("timeliness", "")
        rw = ctx.get("related_works", [])
        if fc: lines3.append(f"**领域背景**：{fc}\n")
        if ws: lines3.append(f"**利害攸关**：{ws}\n")
        if tl: lines3.append(f"**时效性**：{tl}\n")
        if rw: lines3.append(f"*相关文献：{'; '.join(rw[:5])}*\n")
        parts.append("\n".join(lines3))

    return "\n".join(parts)


def _render_figure_arguments_commentary(understanding: dict) -> str:
    """Render figure arguments for commentary type."""
    figs = understanding.get("figure_arguments", [])
    if not figs:
        return ""

    lines = ["## 修辞图示\n"]
    for fig in figs:
        fid = fig.get("figure_id", "Figure ?")
        desc = fig.get("content_description", "")
        rhet = fig.get("rhetorical_function", "")
        loc = fig.get("evidence_location", "")

        loc_tag = f" `{loc}`" if loc else ""
        lines.append(f"### {fid}{loc_tag}\n")
        if desc: lines.append(f"**内容**：{desc}\n")
        if rhet: lines.append(f"**修辞功能**：{rhet}\n")
    return "\n".join(lines)


# ============================================================
# GAPS AND FUTURE DIRECTIONS (review-specific)
# ============================================================

def _render_gaps_and_future(understanding: dict) -> str:
    """Render gaps_and_future section (review/atlas types)."""
    gf = understanding.get("gaps_and_future", {})
    if not gf:
        return ""

    parts = ["## 研究空白与未来方向\n"]

    gaps = gf.get("identified_gaps", [])
    if gaps:
        parts.append("### 已识别研究空白\n")
        for g in gaps:
            gap_text = g.get("gap", "")
            severity = g.get("severity", "")
            loc = g.get("evidence_location", "")
            badge = {"critical": "🔴", "important": "🟡", "minor": "🟢"}.get(severity, "")
            loc_tag = f" `{loc}`" if loc else ""
            parts.append(f"- {badge} **[{severity}]** {gap_text}{loc_tag}")
        parts.append("")

    futures = gf.get("future_directions", [])
    if futures:
        parts.append("### 未来研究方向\n")
        for f in futures:
            direction = f.get("direction", "")
            feasibility = f.get("feasibility", "")
            advances = f.get("required_advances", "")
            badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(feasibility, "")
            parts.append(f"- {badge} **{direction}**")
            if advances:
                parts.append(f"  *所需突破：{advances}*")
        parts.append("")

    return "\n".join(parts)


# ============================================================
# LITERATURE REPORT (universal — comprehensive synthesis)
# ============================================================

def _render_literature_report(understanding: dict) -> str:
    """Render literature_report section (present in all types, rich synthesis)."""
    lr = understanding.get("literature_report")
    if not lr:
        return ""

    parts = ["## 文献综合脉络\n"]

    field_labels = [
        ("topic_summary", "主题概要"),
        ("scope_and_methodology", "范围与方法论"),
        ("thematic_synthesis", "主题综合"),
        ("comparative_analysis", "对比分析"),
        ("gaps_synthesis", "空白综合"),
        ("future_outlook", "未来展望"),
        ("relevance_to_field", "领域意义"),
    ]

    for key, label in field_labels:
        val = lr.get(key)
        if val:
            parts.append(f"### {label}\n")
            parts.append(f"{val}\n")

    return "\n".join(parts)


# ============================================================
# KNOWLEDGE EXTRACTION HINTS (type-aware Phase B guidance)
# ============================================================

# Knowledge subtype definitions per doc_type
KNOWLEDGE_SUBTYPES = {
    "article": [
        ("mechanism", "机制发现", "X通过Y机制实现Z"),
        ("method-innovation", "方法创新", "X方法的创新点在于Y，解决了Z问题"),
        ("empirical-finding", "实证发现", "在X条件下观察到Y，证据强度Z"),
        ("protocol", "可复现方案", "使用X工具(v版本)，参数A=B，适用场景C"),
    ],
    "review": [
        ("concept-definition", "核心概念", "X是指Y，核心特征Z，当前理解水平W"),
        ("field-consensus", "领域共识", "关于X，领域已达成共识Y（证据强度Z）"),
        ("field-debate", "领域争议", "X存在争议：A方 vs B方，当前状态W"),
        ("method-landscape", "方法族图谱", "解决X的方法族有Y、Z，各适用场景W"),
        ("knowledge-evolution", "知识演进", "对X的理解从A演进到B，关键转折C"),
        ("open-question", "开放问题", "X领域未解决Y（可行性Z，所需突破W）"),
    ],
    "benchmark": [
        ("tool-evaluation", "工具评测卡", "X工具(v版本)：优势A，劣势B，最佳场景C"),
        ("method-selection", "方法选型决策", "当条件X时，选Y优于Z，因为W"),
        ("benchmark-methodology", "评测方法论", "评测X类算法应使用Y指标+Z数据，注意W"),
        ("performance-baseline", "性能基线", "X任务SOTA：Y方法在Z上达W性能"),
    ],
    "atlas": [
        ("resource-specification", "资源规格卡", "X数据集：物种A，技术B，规模C，获取D"),
        ("reuse-guide", "复用指南", "用X数据做Y分析：推荐子集Z，注意偏倚W"),
        ("annotation-quality", "注释质量评估", "X注释：策略Y，深度Z，验证W，置信V"),
        ("atlas-finding", "图谱发现", "从X图谱发现Y（新发现程度Z）"),
    ],
    "commentary": [
        ("argument-position", "论点立场", "作者论证X：核心论据Y，证据类型Z"),
        ("field-prediction", "领域预判", "预测X领域将向Y发展，依据Z"),
        ("methodological-critique", "方法论批评", "批评X范式：理由Y，建议替代Z"),
    ],
}


def _render_extraction_hints(understanding: dict, doc_type: str) -> str:
    """Generate type-aware knowledge extraction roadmap for Phase B.

    This section guides QoderWork Phase B on what knowledge notes to extract,
    what subtype each should be, and which raw staging section to draw from.
    """
    subtypes = KNOWLEDGE_SUBTYPES.get(doc_type, KNOWLEDGE_SUBTYPES["article"])

    lines = [
        "## 知识点提取路线图（Phase B 指引）\n",
        f"> [!info] doc_type = `{doc_type}` — 建议提取 3-5 个知识点，优先覆盖以下子类型\n",
    ]

    lines.append("| knowledge_subtype | 含义 | 模板 |")
    lines.append("|---|---|---|")
    for subtype, label, template in subtypes:
        lines.append(f"| `{subtype}` | {label} | {template} |")
    lines.append("")

    # Type-specific extraction source mapping
    source_map = {
        "article": [
            ("research_threads → 各 stage 的 evidence_points", "mechanism / empirical-finding"),
            ("core_methods → 工具+参数+用途", "protocol / method-innovation"),
            ("figure_arguments → sub_argument + weakness", "empirical-finding"),
            ("key_findings → 高 strength 条目", "empirical-finding"),
        ],
        "review": [
            ("thematic_threads → 各 theme 的 current_state + open_questions", "concept-definition / open-question"),
            ("consensus_and_debate → consensus_points", "field-consensus"),
            ("consensus_and_debate → debate_points", "field-debate"),
            ("methods_landscape → 各 method_family", "method-landscape"),
            ("thematic_threads → evolution_narrative", "knowledge-evolution"),
            ("gaps_and_future → future_directions", "open-question"),
        ],
        "benchmark": [
            ("rankings_and_recommendations → overall_ranking 每行", "tool-evaluation"),
            ("evaluation_threads → surprising_results + confounding", "benchmark-methodology"),
            ("benchmark_design → evaluation_protocol + ground_truth", "benchmark-methodology"),
            ("methodological_insights → what_matters + parameter_sensitivity", "method-selection"),
            ("rankings_and_recommendations → practical_guidelines", "method-selection"),
        ],
        "atlas": [
            ("resource_design → technology_platform + sample_design", "resource-specification"),
            ("data_accessibility → repositories + formats + license", "resource-specification"),
            ("annotation_assessment → strategy + validation + confidence", "annotation-quality"),
            ("initial_biological_findings → 高 novelty 条目", "atlas-finding"),
            ("research_recommendations → recommended_subset + analysis_suggestions", "reuse-guide"),
        ],
        "commentary": [
            ("commentary_structure → argument_chain 各步骤", "argument-position"),
            ("commentary_structure → main_position + implications_if_correct", "field-prediction"),
            ("context_and_stakes → field_context + what_is_at_stake", "field-prediction"),
            ("commentary_structure → counterarguments_acknowledged", "methodological-critique"),
        ],
    }

    sources = source_map.get(doc_type, source_map["article"])
    lines.append("### 提取来源映射\n")
    lines.append("| 原始数据路径 | 建议 subtype |")
    lines.append("|---|---|")
    for src, target in sources:
        lines.append(f"| {src} | `{target}` |")
    lines.append("")

    # Phase B action instruction
    lines += [
        "### Phase B 操作指令\n",
        "1. 按上表优先级从高到低扫描各节内容",
        "2. 每个知识点应**原子化**（独立可理解，不依赖其他点）",
        "3. frontmatter 中标注 `knowledge_subtype: <subtype>`",
        "4. 知识点命名（SLUG）反映核心概念，而非论文名",
        "5. 如多个 section 指向同一知识点，合并为一个更丰富的笔记",
        "6. `出处` 字段链接本文 source card",
        "",
    ]

    return "\n".join(lines)


# ============================================================
# RESEARCH RECOMMENDATIONS (type-aware)
# ============================================================

def _render_research_recommendations(understanding: dict, doc_type: str) -> str:
    """Render research_recommendations with type-aware section labels."""
    rec = understanding.get("research_recommendations")
    if not rec:
        return ""

    # Map doc_type → field:label pairs
    field_map = {
        "article": [
            ("borrowable_ideas", "可借鉴思路"),
            ("supplementary_analyses", "补充分析"),
            ("reproduction_path", "复现路径"),
            ("caveats_and_limitations", "注意事项"),
            ("migration_suggestions", "迁移建议"),
        ],
        "review": [
            ("knowledge_gaps_for_user", "知识空白"),
            ("starting_points", "入门起点"),
            ("positioning_advice", "定位建议"),
            ("caveats", "综述局限性"),
        ],
        "benchmark": [
            ("method_selection_for_user", "方法选型建议"),
            ("pipeline_integration", "管线集成"),
            ("missing_evaluations", "遗漏评测维度"),
            ("reproduction_path", "复现路径"),
        ],
        "atlas": [
            ("how_to_access", "获取数据"),
            ("recommended_subset", "推荐子集"),
            ("analysis_suggestions", "分析建议"),
            ("integration_strategy", "整合策略"),
            ("caveats_for_reuse", "使用注意"),
        ],
        "commentary": [
            ("position_assessment", "立场评估"),
            ("implications_for_user_work", "对在研方向的影响"),
            ("follow_up_readings", "跟进阅读"),
        ],
    }

    fields = field_map.get(doc_type, field_map["article"])
    lines = ["## 研究建议\n"]
    has_any = False
    for key, label in fields:
        val = rec.get(key)
        if not val:
            continue
        has_any = True
        if isinstance(val, list):
            lines.append(f"**{label}**：")
            for item in val:
                lines.append(f"- {item}")
        else:
            lines.append(f"**{label}**：{val}\n")

    return "\n".join(lines) + "\n" if has_any else ""


# ============================================================
# RELEVANCE TO USER (type-aware)
# ============================================================

def _render_relevance(understanding: dict, doc_type: str) -> str:
    """Render relevance_to_user with type-aware sections."""
    rtu = understanding.get("relevance_to_user")
    if not rtu:
        return ""

    lines = ["## 个性化解读\n"]
    has_any = False

    # Pain points (universal)
    pp = rtu.get("pain_point_matches", [])
    if pp:
        has_any = True
        lines.append("### 痛点匹配\n")
        for p in pp:
            ppt = p.get("user_pain_point", "")
            addr = p.get("paper_addresses", "") or p.get("resource_addresses", "")
            action = p.get("actionability", "")
            badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(action, "")
            lines.append(f"- {badge} **{ppt}**：{addr}")
        lines.append("")

    # Type-specific fields
    type_fields = {
        "article": [
            ("transferable_methods", "可迁移方法"),
            ("novel_insights_for_user", "新颖洞见"),
        ],
        "review": [
            ("recommended_readings", "推荐阅读"),
            ("method_recommendations", "方法推荐"),
            ("knowledge_map_value", "知识图谱价值"),
        ],
        "benchmark": [
            ("directly_applicable_methods", "可直接采用的方法"),
            ("datasets_reusable", "可复用数据集"),
            ("parameter_recommendations", "参数建议"),
        ],
        "atlas": [
            ("reuse_scenarios", "复用场景"),
            ("integration_with_user_data", "数据整合建议"),
            ("complementary_to_user_work", "互补价值"),
        ],
        "commentary": [
            ("perspective_value", "视角启发"),
            ("action_items", "行动事项"),
        ],
    }

    fields = type_fields.get(doc_type, type_fields["article"])
    for key, label in fields:
        val = rtu.get(key)
        if not val:
            continue
        has_any = True
        if isinstance(val, list):
            lines.append(f"### {label}\n")
            for item in val:
                lines.append(f"- {item}")
        else:
            lines.append(f"**{label}**：{val}\n")
        lines.append("")

    return "\n".join(lines) + "\n" if has_any else ""


# ============================================================
# RAW STAGING FILE BUILDER (v4.1 — multi-type)
# ============================================================

def build_raw_staging(output_data: dict, zotero_key: str, theme_path: str) -> tuple[str, str]:
    """
    Build a full analysis raw staging file for raw/.
    Phase B will pick this up, extract 3-5 knowledge notes, and archive it.
    Routes to type-specific renderers based on doc_type.
    Returns (content, filename).
    """
    understanding = output_data.get("understanding", {})
    concepts = understanding.get("concepts", [])
    doc_type = _resolve_doc_type(output_data)

    title = understanding.get("title", "Untitled")
    short_name = _make_short_name(title)
    authors = understanding.get("authors", "")
    journal_year = understanding.get("journal_year", "")
    doi = understanding.get("doi", "")
    today = date.today().isoformat()
    analysis_mode = output_data.get("analysis_mode", "overview")
    contribution_type = understanding.get("paper_contribution_type", "")

    # Tags
    tags_str = ", ".join(concepts[:8]) if concepts else ""

    # Escape quotes
    title_yaml = title.replace('"', '\\"')

    # --- Build frontmatter ---
    fm_lines = [
        "---",
        f'zotero_key: "{zotero_key}"',
        f'title: "{title_yaml}"',
        f'authors: ["{authors}"]',
        f'journal_year: "{journal_year}"',
    ]
    if doi:
        fm_lines.append(f'doi: "{doi}"')
    if theme_path:
        fm_lines.append(f'themes: ["{theme_path}"]')
    if tags_str:
        fm_lines.append(f"tags: [{tags_str}]")
    if contribution_type:
        fm_lines.append(f"paper_contribution_type: {contribution_type}")
    if doc_type != "article":
        fm_lines.append(f"doc_type: {doc_type}")
    fm_lines += [
        f'hlokk_analyzed: "{today}"',
        f'analysis_mode: "{analysis_mode}"',
        'status: "raw"',
        "---",
    ]

    frontmatter = "\n".join(fm_lines)

    # --- Route to type-specific renderers ---
    # Universal sections
    universal_sections = [
        frontmatter,
        "",
        _render_core_problem(understanding, doc_type),
        _render_key_findings(understanding),
        _render_reading_cipher(understanding),
        _render_limitations(understanding, doc_type),
        _render_relevance(understanding, doc_type),
        _render_research_recommendations(understanding, doc_type),
        _render_literature_report(understanding),
    ]

    # Type-specific sections
    if doc_type == "review":
        type_sections = [
            _render_narrative_review(understanding),
            _render_methods_review(understanding),
            _render_figure_arguments_review(understanding),
            _render_gaps_and_future(understanding),
        ]
    elif doc_type == "benchmark":
        type_sections = [
            _render_narrative_benchmark(understanding),
            _render_methods_benchmark(understanding),
            _render_figure_arguments_benchmark(understanding),
            _render_gaps_and_future(understanding),
        ]
    elif doc_type == "atlas":
        type_sections = [
            _render_narrative_atlas(understanding),
            _render_methods_atlas(understanding),
            _render_figure_arguments_atlas(understanding),
            _render_gaps_and_future(understanding),
        ]
    elif doc_type == "commentary":
        type_sections = [
            _render_narrative_commentary(understanding),
            _render_figure_arguments_commentary(understanding),
        ]
    else:  # article
        type_sections = [
            _render_narrative_map_article(understanding),
            _render_methods_article(understanding),
            _render_figure_arguments_article(understanding),
        ]

    sections = universal_sections + type_sections

    # Add concepts as a structured section for Phase B extraction
    if concepts:
        sections.append("## 相关概念\n")
        sections.append(", ".join(concepts))
        sections.append("")

    # Add type-aware extraction hints for Phase B
    sections.append(_render_extraction_hints(understanding, doc_type))

    content = "\n".join(s for s in sections if s.strip())
    filename = f"{zotero_key}_{short_name}.md"

    return content, filename


# ============================================================
# VAULT WRITER (v4.0)
# ============================================================

def write_source_card(content: str, filename: str, vault_path: str) -> Path:
    """Write source card to Sources/ directory."""
    vault = Path(vault_path)
    if not vault.exists():
        print(f"ERROR: Vault path does not exist: {vault}", file=sys.stderr)
        sys.exit(1)

    dest_dir = vault / "Sources"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_file = dest_dir / filename

    # Handle filename collision: if same zotero_key, overwrite; otherwise suffix
    if dest_file.exists():
        existing = dest_file.read_text(encoding="utf-8")
        # Check if same paper (by zotero_key in new content)
        zk_match = re.search(r'zotero_key:\s*(\S+)', content)
        if zk_match and zk_match.group(1) in existing:
            pass  # Same paper, overwrite
        else:
            base = dest_file.stem
            suffix = 1
            while dest_file.exists():
                dest_file = dest_dir / f"{base}-{suffix}.md"
                suffix += 1

    dest_file.write_text(content, encoding="utf-8")
    return dest_file


def write_raw_staging(content: str, filename: str, vault_path: str) -> Path | None:
    """Write full analysis to raw/ directory for Phase B knowledge extraction."""
    raw_dir = Path(vault_path) / "raw"
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)

    dest_file = raw_dir / filename
    dest_file.write_text(content, encoding="utf-8")
    return dest_file


# ============================================================
# PROCESSED.JSON 更新 (v4.1)
# ============================================================

def _update_processed_json(
    vault: Path,
    zotero_key: str,
    title: str,
    doi: str,
    theme_path: str,
    source_card_path: str,
    raw_staging_path: str,
    doc_type: str = "article",
) -> None:
    """Update _meta/processed.json with the new source card record (v4.1 multi-type)."""
    processed_path = vault / "_meta" / "processed.json"
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if processed_path.exists():
        try:
            data = json.loads(processed_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}

    dt_config = DOC_TYPE_CONFIG.get(doc_type, DOC_TYPE_CONFIG["article"])

    data[zotero_key] = {
        "type": dt_config["card_type"],
        "doc_type": doc_type,
        "date_processed": date.today().isoformat(),
        "zotero_modified": "",
        "themes": [theme_path],
        "path": source_card_path,
        "raw_staging": raw_staging_path,
        "knowledge_notes": [],  # Phase B will populate this
        "title": title,
        "doi": doi,
    }

    processed_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# LEGACY COMPATIBILITY: build_entity_page (kept for backward compat)
# ============================================================

def build_entity_page(output_data: dict, zotero_key: str, theme_path: str) -> tuple[str, str]:
    """Legacy wrapper — now generates source card instead of entity page."""
    return build_source_card(output_data, zotero_key, theme_path)


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert Hlokk JSON output to HlokkObsidian v4.1 (multi-type source card + raw staging)"
    )
    parser.add_argument("--json", required=True, help="Path to Hlokk JSON output file")
    parser.add_argument(
        "--vault", default=None,
        help="Path to HlokkObsidian vault. If omitted, output to same directory as JSON."
    )
    parser.add_argument(
        "--zotero-key", default=None, dest="zotero_key",
        help="Zotero item key (e.g., ABC12345)"
    )
    parser.add_argument(
        "--theme", default=None,
        help="HlokkObsidian theme path (e.g., 'Genomics/Data-Analysis'). "
             f"Default: '{DEFAULT_THEME}'"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the generated content to stdout without writing to disk"
    )
    args = parser.parse_args()

    # Load JSON
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"ERROR: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        output_data = json.load(f)

    understanding = output_data.get("understanding", {})
    doc_type = _resolve_doc_type(output_data)

    # Resolve zotero_key: CLI arg > embedded in JSON > None
    zotero_key = args.zotero_key or output_data.get("zotero_key") or ""

    # Resolve theme
    theme_path = args.theme or DEFAULT_THEME

    # Build source card + raw staging
    card_content, card_filename = build_source_card(output_data, zotero_key, theme_path)
    raw_content, raw_filename = build_raw_staging(output_data, zotero_key, theme_path)

    if args.dry_run:
        dt_config = DOC_TYPE_CONFIG.get(doc_type, DOC_TYPE_CONFIG["article"])
        print(f"[doc_type: {doc_type} ({dt_config['label']})]")
        print("=" * 60)
        print("SOURCE CARD (Sources/)")
        print("=" * 60)
        print(card_content)
        print("\n" + "=" * 60)
        print("RAW STAGING (raw/)")
        print("=" * 60)
        print(raw_content)
        return

    # Determine vault path
    vault_path = args.vault
    if not vault_path:
        config_candidates = [
            json_path.parent.parent / "config.yaml",
            json_path.parent / "config.yaml",
        ]
        for cfg in config_candidates:
            if cfg.exists():
                try:
                    import yaml
                    cfg_data = yaml.safe_load(cfg.read_text())
                    vault_path = cfg_data.get("vault_path")
                    if vault_path:
                        break
                except Exception:
                    pass

    if not vault_path:
        # Fall back: save alongside the JSON
        out_card = json_path.parent / card_filename
        out_raw = json_path.parent / raw_filename
        out_card.write_text(card_content, encoding="utf-8")
        out_raw.write_text(raw_content, encoding="utf-8")
        print(f"✓ Source card saved (no vault path): {out_card}")
        print(f"✓ Raw staging saved (no vault path): {out_raw}")
        return

    # Write to vault
    card_path = write_source_card(card_content, card_filename, vault_path)
    print(f"✓ Source card: {card_path}")

    raw_path = write_raw_staging(raw_content, raw_filename, vault_path)
    print(f"✓ Raw staging (status: raw): {raw_path}")

    # Update processed.json
    vault = Path(vault_path)
    if zotero_key:
        title = understanding.get("title", "Untitled")
        doi = understanding.get("doi", "")
        card_rel = str(card_path.relative_to(vault))
        raw_rel = str(raw_path.relative_to(vault)) if raw_path else ""
        try:
            _update_processed_json(
                vault, zotero_key, title, doi, theme_path, card_rel, raw_rel,
                doc_type=doc_type,
            )
            print("  processed.json updated")
        except Exception as e:
            print(f"  ⚠️  processed.json update failed: {e}", file=sys.stderr)
    else:
        print("  ⚠️  No Zotero key — add --zotero-key to enable knowledge base linking")


if __name__ == "__main__":
    main()
