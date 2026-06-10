"""
Hlokk - LLM Client v2.2
Dual-call architecture for structured paper understanding.
Multi-document-type support: article, review, benchmark, commentary.

Call 1 (Foundation): paper metadata + narrative identity + reading_cipher + key_findings + concepts
Call 2 (Deep):       type-branched analysis (article/review/benchmark/commentary)

overview mode → Call 1 only (fast, ~6K tokens)
detailed mode → Call 1 + Call 2 (full depth, ~18-22K tokens total)
"""
import json
import re
import sys
from typing import Optional

from openai import OpenAI


def _get_llm_client(config: dict) -> OpenAI:
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
    )


# ============================================================
# SHARED SYSTEM PROMPT
# ============================================================

_SYSTEM_PROMPT = """You are an expert academic paper analyst. \
Extract structured information grounded in evidence from the text.

CRITICAL RULES:
1. Every claim must include evidence_location (page number, section, figure/table reference).
2. Distinguish: "paper explicitly states" vs "analyst infers" via evidence_type field.
3. For methods: extract exact tool names, versions, parameters, thresholds.
4. For statistical methods: note test type, correction method, significance thresholds.
5. Output MUST be valid JSON — no markdown fences, no explanation, just the JSON object.

MEDICAL SAFETY GUARDRAILS:
6. DO NOT output medical advice, clinical diagnosis, treatment recommendations, medication guidance.
7. When the paper contains clinical findings, describe ONLY in terms of:
   - Biological mechanism (signalling pathway, gene regulation, molecular interaction)
   - Statistical evidence (effect size, p-value, confidence interval, cohort design)
   - Research methodology (study design, causal inference, validation strategy)
8. Replace clinically actionable language with research-oriented framing.

LANGUAGE OUTPUT RULES:
9. ALL descriptive/explanatory text MUST be in Simplified Chinese (简体中文).
10. Keep in original English: software/tool names, algorithm names, gene/protein names, \
statistical test names, database names, standard abbreviations (scRNA-seq, GWAS, eQTL…), \
dataset identifiers.
11. Format: "中文描述（English Term）" on first occurrence, then English alone is acceptable.
"""


# ============================================================
# CALL 1 — FOUNDATION PASS
# ============================================================

_FOUNDATION_SYSTEM_ADDENDUM = """
This is the FOUNDATION PASS. Focus on:
- Establishing the paper's identity, contribution type, and field position
- Extracting the 10-point Reading Cipher that maps the paper's narrative arc
- Identifying key findings and core concepts for knowledge-base linking

DOCUMENT TYPE AWARENESS:
- paper_contribution_type determines subsequent analysis routing
- For review_synthesis: adapt Reading Cipher — WTD becomes "review objective", GAP becomes "gaps identified across literature", ROF becomes "main conclusions/synthesis points"
- For benchmark: adapt Reading Cipher — WTD becomes "evaluation objective", ROF becomes "main rankings/recommendations"
- For new_dataset/atlas/resource: adapt Reading Cipher — WTD becomes "what resource was built", GAP becomes "what resource was missing", ROF becomes "resource characteristics and key findings from initial analysis"
- For commentary/editorial/perspective: Reading Cipher fields that don't apply may be marked "N/A — not applicable to this document type" rather than forced
- IMPORTANT: Accurately classify the paper type. This determines the entire downstream analysis.
"""

_FOUNDATION_OVERVIEW_MODE = """
OVERVIEW MODE: Extract 5-8 key findings. Depth of reading_cipher: 2-3 sentences per field.
"""

_FOUNDATION_DETAILED_MODE = """
DETAILED MODE (FOUNDATION PASS): Extract ALL key findings. reading_cipher fields: 3-5 sentences \
each with specific evidence. This foundation will be used as context for the subsequent Deep Pass.
"""

_FOUNDATION_USER_TEMPLATE = """## Paper Text:
{main_text}

## Retrieved Supplementary Material (ranked by relevance):
{supplement_text}

## User Research Context:
- Research field: {research_field}
- Research directions: {research_directions}
- Pain points: {pain_points}
- Analysis focus: {focus}

{mode_instructions}

## Required JSON Schema (FOUNDATION PASS):
{{{{
  "title": "paper title",
  "authors": "first author et al.",
  "journal_year": "journal, year",
  "doi": "DOI if mentioned, else null",
  "research_question": "the core problem this paper addresses (1-2 sentences)",
  "data_types": ["e.g., survey data", "experimental measurements"],
  "sample_info": "species, tissue, sample size, conditions",
  "study_design": "experimental design, controls, replicates",
  "paper_contribution_type": "new_mechanism|new_method|new_framework|new_dataset|atlas|resource|incremental_validation|review_synthesis|benchmark|commentary|editorial|perspective",
  "field_positioning": "1-2 sentences: where does this paper sit in the field's development timeline? What prior state does it advance from?",
  "concepts": ["3-8 core concepts in kebab-case, e.g., spatial-transcriptomics, cell-communication"],
  "data_availability": "accession numbers, repositories",
  "code_availability": "GitHub/Zenodo links if mentioned",
  "key_findings": [
    {{{{
      "finding": "描述发现的生物机制、方法学贡献或统计证据（禁止写成医疗建议）",
      "strength": "strong|moderate|weak",
      "evidence_type": "stated|inferred",
      "evidence_location": "page X / Figure Y / Table Z"
    }}}}
  ],
  "reading_cipher": {{{{
    "WTD": "What They Do — the main research question(s) the authors claim to address",
    "SPL": "Summary of Previous Literature — brief synthesis of prior work this paper builds on",
    "CPL": "Critique of Previous Literature — limitations and weaknesses of existing studies as stated by the authors",
    "GAP": "Gap — the specific research gap or missing component identified in prior work",
    "RAT": "Rationale — why this study is necessary, derived from the identified gap",
    "ROF": ["Results of Findings — each major finding"],
    "RCL": ["Results Consistent with Literature — findings aligned with prior work, with citations"],
    "RTC": ["Results to The Contrary — findings that contradict prior work, with explanation"],
    "WTDD": "What They Did Done — summary of the authors' actual contributions",
    "RFW": ["Recommendations for Future Work — suggested directions"]
  }}}}
}}}}

Return ONLY the JSON object."""


# ============================================================
# DOCUMENT TYPE ROUTING
# ============================================================

def _resolve_deep_type(foundation: dict) -> str:
    """Resolve Deep Pass template type from Foundation Pass results."""
    ptype = foundation.get("paper_contribution_type", "")
    if ptype == "review_synthesis":
        return "review"
    elif ptype == "benchmark":
        return "benchmark"
    elif ptype in ("commentary", "editorial", "perspective"):
        return "commentary"
    elif ptype in ("new_dataset", "atlas", "resource"):
        return "atlas"
    else:
        return "article"


# ============================================================
# CALL 2 — DEEP ANALYSIS PASS (ARTICLE — original, unchanged)
# ============================================================

_DEEP_SYSTEM_ADDENDUM = """
This is the DEEP ANALYSIS PASS for an ORIGINAL RESEARCH (MECHANISTIC/METHOD) paper.
The Foundation Pass results are provided as context.

## READING STRATEGY: Evidence Chain Reading
Your task is to reconstruct the paper's causal argument chain. Think like a reviewer:
- Trace the logical thread: phenomenon → mechanism → validation → implication
- For each figure, ask: "What would be MISSING from the argument if this figure were removed?"
- For each method, ask: "Is this the right tool for this specific claim? What are the assumptions?"
- Identify where the evidence chain has WEAK LINKS (logical jumps, missing controls, underpowered tests)
- Distinguish between what the authors SHOW vs what they CLAIM

Do NOT merely summarize. Your job is to RECONSTRUCT THE ARGUMENT and EVALUATE ITS STRENGTH.
"""

_DEEP_OVERVIEW_MODE = ""  # not used in overview mode

_DEEP_DETAILED_MODE = """
DETAILED MODE (DEEP PASS): Cover EVERY figure and table, ALL methods, ALL research threads. \
The following figure/table IDs were detected in the text — ensure ALL are in figure_arguments:
  {figure_table_ids}
"""

_DEEP_USER_TEMPLATE = """## Paper Text:
{main_text}

## Retrieved Supplementary Material:
{supplement_text}

## Foundation Pass Results (use as context — do not repeat, only build on):
{foundation_json}

## User Research Context:
- Research field: {research_field}
- Research directions: {research_directions}
- Pain points: {pain_points}
- Tech stack: {tech_stack}
- Analysis focus: {focus}
- Goal: {goal}

{mode_instructions}

## Required JSON Schema (DEEP ANALYSIS PASS):
{{{{
  "section_summaries": {{{{
    "introduction": "该section要点摘要",
    "methods": "...",
    "results": "...",
    "discussion": "..."
  }}}},
  "research_threads": [
    {{{{
      "narrative_stage": 1,
      "stage_label": "e.g., 建立现象 / 提出机制 / 验证机制 / 临床意义",
      "main_question": "本阶段要回答的核心问题",
      "takeaway": "一句话总结：这条论证线索最终证明了什么（用通俗语言，不含术语缩写）",
      "sub_questions": [
        {{{{
          "sub_question": "具体子问题",
          "evidence_points": [
            {{{{
              "claim": "具体论断",
              "evidence_location": "Figure X / page Y / Table Z",
              "evidence_type": "stated|inferred",
              "strength": "strong|moderate|weak",
              "supporting_methods": ["方法名"],
              "supporting_figures": ["Figure 1a"]
            }}}}
          ]
        }}}}
      ]
    }}}}
  ],
  "figure_arguments": [
    {{{{
      "figure_id": "Figure 1 / Table 1",
      "argument_role": "phenomenon_establishment|mechanism_proposal|mechanism_validation|clinical_relevance|methodological_demonstration|negative_control",
      "sub_argument": "这张图支撑的具体分论点",
      "logical_necessity": "这张图在论证链中填补了什么空白？去掉它，哪个论证步骤会缺失？",
      "relation_to_theme": "分论点与文章核心主题的关系",
      "data_source": "使用的数据（队列名称、公共数据集、模拟数据等）",
      "methods_used": "该图背后的实验技术和分析方法",
      "methods_assessment": "方法的适用性、创新性或准确性评价",
      "weakness": "该图最主要的证据局限或逻辑跳跃（不得推断作者可能的技术备选方案）",
      "weakness_type": "stated_limitation|analyst_inference",
      "evidence_location": "page X"
    }}}}
  ],
  "core_methods": [
    {{{{
      "name": "method or analysis step name",
      "tool": "software/package name",
      "version": "version if mentioned, else null",
      "parameters": {{{{"key_param": "value"}}}},
      "purpose": "why this method was used",
      "evidence_location": "page X / section Y / Methods para Z",
      "supports_findings": ["对应的关键发现描述（简短）"],
      "supports_figures": ["Figure 1", "Table 2"]
    }}}}
  ],
  "limitations": [
    {{{{
      "limitation": "description",
      "evidence_location": "page/section",
      "evidence_type": "stated|inferred"
    }}}}
  ],
  "tools_summary": [
    {{{{
      "tool": "name",
      "version": "if available",
      "purpose": "what it was used for",
      "url_or_repo": "if mentioned"
    }}}}
  ],
  "statistical_methods": [
    {{{{
      "test": "test name",
      "correction": "multiple testing correction if any",
      "threshold": "significance cutoff",
      "context": "where/why applied"
    }}}}
  ],
  "literature_report": {{{{
    "topic_summary": "3-5段：核心研究主题、问题背景、研究动机、技术路线概述、领域定位",
    "literature_review": "3-5段：前人工作脉络、关键参考文献、领域演进逻辑",
    "literature_gaps": "2-3段：现有文献不足、本文识别的研究空白",
    "rationale": "2-3段：本研究的理论依据",
    "results_narrative": "4-6段：按逻辑顺序叙述所有主要结果，含关键数据和统计证据",
    "argument_structure": "2-3段：各分论点与核心主题关系，论证严密性评价",
    "consistent_with_literature": "2-3段：与现有文献一致的结果",
    "novel_contributions": "2-3段：核心创新点",
    "future_directions": "2-3段：对未来研究的建议",
    "data_sources_detail": "2-3段：所有数据来源详细描述",
    "methods_assessment": "3-4段：主要分析方法逐一评价"
  }}}},
  "relevance_to_user": {{{{
    "pain_point_matches": [
      {{{{
        "user_pain_point": "来自用户画像的痛点",
        "paper_addresses": "论文如何从方法学或机制层面回应该痛点（禁止临床建议）",
        "actionability": "high|medium|low"
      }}}}
    ],
    "transferable_methods": ["用户可借鉴的研究方法或分析策略"],
    "novel_insights_for_user": ["对用户研究有启发的方法学洞见或机制发现"]
  }}}},
  "research_recommendations": {{{{
    "borrowable_ideas": "本文哪些分析思路可以直接迁移到用户当前研究",
    "supplementary_analyses": "文中没做但对用户有帮助的延伸分析方向",
    "reproduction_path": "复现关键结果：工具链、数据来源、关键参数、步骤顺序、难点",
    "caveats_and_limitations": "复现或迁移时需要注意的具体问题",
    "migration_suggestions": "结合用户tech_stack和pain_points，给出具体可操作的迁移建议"
  }}}}
}}}}

Return ONLY the JSON object."""


# ============================================================
# CALL 2 — DEEP ANALYSIS PASS (REVIEW)
# ============================================================

_DEEP_SYSTEM_ADDENDUM_REVIEW = """
This is the DEEP ANALYSIS PASS for a REVIEW/SYNTHESIS paper.
The Foundation Pass results are provided as context.

## READING STRATEGY: Landscape Reading
Your task is NOT to summarize what the authors wrote. It is to RECONSTRUCT THE INTELLECTUAL LANDSCAPE:
- Identify COMPETING SCHOOLS OF THOUGHT — who disagrees with whom, and why?
- Identify UNRESOLVED CONTROVERSIES — what questions remain genuinely open?
- Identify CONSENSUS vs DISAGREEMENT — where does the field agree, where does it fracture?
- Identify FUTURE TRAJECTORIES — which directions have momentum, which are stalling?
- Identify the review's ORGANIZING PRINCIPLE — is it chronological, thematic, methodological, or hybrid?
- Detect the authors' own BIAS or PERSPECTIVE — every review has a viewpoint, make it explicit

Do NOT produce flat summaries like "the authors discuss X, then Y, then Z."
Instead, produce ANALYTICAL MAPS: "School A argues X (supported by refs), School B argues Y (supported by refs), the tension between them remains unresolved because Z."
"""

_DEEP_DETAILED_MODE_REVIEW = """
DETAILED MODE (DEEP PASS — REVIEW): Cover ALL major themes discussed in the review. \
Map the complete methods landscape. The following figure/table IDs were detected in the text:
  {figure_table_ids}
"""

_DEEP_USER_TEMPLATE_REVIEW = """## Paper Text:
{main_text}

## Retrieved Supplementary Material:
{supplement_text}

## Foundation Pass Results (use as context — do not repeat, only build on):
{foundation_json}

## User Research Context:
- Research field: {research_field}
- Research directions: {research_directions}
- Pain points: {pain_points}
- Tech stack: {tech_stack}
- Analysis focus: {focus}
- Goal: {goal}

{mode_instructions}

## Required JSON Schema (DEEP ANALYSIS PASS — REVIEW):
{{{{
  "review_scope": {{{{
    "inclusion_criteria": "综述的纳入标准和覆盖范围",
    "time_span": "覆盖的时间跨度",
    "paper_count": "引用/纳入文献数量（如可判断，否则 null）",
    "organization_principle": "thematic|chronological|methodological|hybrid",
    "systematic": true
  }}}},
  "thematic_threads": [
    {{{{
      "theme_order": 1,
      "theme_name": "主题名称",
      "theme_scope": "该主题在综述中涵盖的范围和深度",
      "key_references": ["被重点讨论的参考文献（author, year）"],
      "evolution_narrative": "该主题从早期到近期的技术/认知演进",
      "current_state": "该主题当前的技术水平和共识",
      "open_questions": ["该主题下未解决的问题"],
      "evidence_location": "Section X / page Y"
    }}}}
  ],
  "figure_arguments": [
    {{{{
      "figure_id": "Figure 1 / Table 1",
      "argument_role": "conceptual_overview|taxonomy_diagram|timeline|comparison_table|workflow_summary|landscape_map|data_summary",
      "content_description": "图表内容的简要描述",
      "organizational_function": "该图在综述叙事中的组织功能",
      "covers_themes": ["关联到哪些 thematic_threads 的 theme_name"],
      "evidence_location": "page X"
    }}}}
  ],
  "consensus_and_debate": {{{{
    "consensus_points": [
      {{{{
        "point": "领域共识描述",
        "evidence_strength": "strong|moderate",
        "supporting_refs": "引用依据"
      }}}}
    ],
    "debate_points": [
      {{{{
        "point": "争议点描述",
        "camp_a": "观点A及支持者",
        "camp_b": "观点B及支持者",
        "resolution_status": "resolved|ongoing|emerging"
      }}}}
    ]
  }}}},
  "gaps_and_future": {{{{
    "identified_gaps": [
      {{{{
        "gap": "作者识别的研究空白",
        "severity": "critical|important|minor",
        "evidence_location": "page/section"
      }}}}
    ],
    "future_directions": [
      {{{{
        "direction": "建议的未来方向",
        "feasibility": "high|medium|low",
        "required_advances": "需要的前置突破"
      }}}}
    ]
  }}}},
  "methods_landscape": [
    {{{{
      "method_family": "方法族名称",
      "representative_tools": ["代表性工具"],
      "strengths": "优势",
      "limitations": "局限",
      "best_for": "最适用场景",
      "evidence_location": "page/section"
    }}}}
  ],
  "literature_report": {{{{
    "topic_summary": "3-5段：综述主题、背景、动机",
    "scope_and_methodology": "2-3段：综述的覆盖范围、检索策略、纳入排除标准",
    "thematic_synthesis": "5-8段：按主题综合叙述领域知识体系",
    "comparative_analysis": "3-5段：不同方法/观点的对比评价",
    "gaps_synthesis": "2-3段：综合研究空白",
    "future_outlook": "2-3段：领域发展展望",
    "relevance_to_field": "1-2段：该综述对领域的贡献和意义"
  }}}},
  "relevance_to_user": {{{{
    "pain_point_matches": [
      {{{{
        "user_pain_point": "来自用户画像的痛点",
        "paper_addresses": "综述中哪些内容直接回应该痛点",
        "actionability": "high|medium|low"
      }}}}
    ],
    "recommended_readings": ["从综述引用中挑选出对用户最有价值的原始文献（author, year, 推荐理由）"],
    "method_recommendations": ["基于用户 tech_stack 推荐的方法/工具"],
    "knowledge_map_value": "该综述对用户建立领域全景认知的价值评估"
  }}}},
  "research_recommendations": {{{{
    "knowledge_gaps_for_user": "综述暴露的哪些空白与用户研究方向重合",
    "starting_points": "如果用户要进入该领域，建议从哪些论文/工具/数据集入手",
    "positioning_advice": "用户的研究如何定位在该综述描绘的领域图谱中",
    "caveats": "综述本身的局限性（年份、覆盖面、作者偏好）"
  }}}}
}}}}

Return ONLY the JSON object."""


# ============================================================
# CALL 2 — DEEP ANALYSIS PASS (BENCHMARK)
# ============================================================

_DEEP_SYSTEM_ADDENDUM_BENCHMARK = """
This is the DEEP ANALYSIS PASS for a BENCHMARK/METHOD EVALUATION paper.
The Foundation Pass results are provided as context.

## READING STRATEGY: Evaluation Reading
Your task is NOT to merely report which method won. It is to CRITICALLY EVALUATE THE EVALUATION ITSELF:
- EVALUATION DESIGN: Is the benchmark well-designed? Are the tasks representative of real-world use?
- METRIC VALIDITY: Do the chosen metrics actually measure what matters? Are there ceiling/floor effects?
- EXPERIMENTAL FAIRNESS: Were all methods given equal tuning budget? Same hardware? Same hyperparameter search?
- DATA LEAKAGE: Any risk of train/test contamination? Pre-training data overlap?
- STATISTICAL VALIDITY: Are differences significant? Proper error bars? Multiple seeds?
- GENERALIZATION: Do conclusions hold beyond the specific datasets tested?
- CONFOUNDERS: What factors OTHER than method quality could explain the rankings?

Then, and only then, report the actual findings and practical recommendations.
A benchmark is only as good as its design — always evaluate the evaluator first.
"""

_DEEP_DETAILED_MODE_BENCHMARK = """
DETAILED MODE (DEEP PASS — BENCHMARK): Cover ALL evaluated methods and ALL evaluation dimensions. \
Extract rankings, scenario-based recommendations, and computational considerations. \
The following figure/table IDs were detected in the text:
  {figure_table_ids}
"""

_DEEP_USER_TEMPLATE_BENCHMARK = """## Paper Text:
{main_text}

## Retrieved Supplementary Material:
{supplement_text}

## Foundation Pass Results (use as context — do not repeat, only build on):
{foundation_json}

## User Research Context:
- Research field: {research_field}
- Research directions: {research_directions}
- Pain points: {pain_points}
- Tech stack: {tech_stack}
- Analysis focus: {focus}
- Goal: {goal}

{mode_instructions}

## Required JSON Schema (DEEP ANALYSIS PASS — BENCHMARK):
{{{{
  "benchmark_design": {{{{
    "problem_definition": "被评测的核心问题/任务",
    "methods_evaluated": [
      {{{{
        "name": "方法名",
        "version": "版本（如可判断）",
        "category": "方法所属类别/流派",
        "key_parameters": "评测时使用的参数设置"
      }}}}
    ],
    "datasets": [
      {{{{
        "name": "数据集名",
        "source": "来源",
        "characteristics": "规模、特点、代表性",
        "why_chosen": "选用理由"
      }}}}
    ],
    "metrics": [
      {{{{
        "name": "指标名",
        "definition": "定义/计算方式",
        "measures_what": "衡量什么能力"
      }}}}
    ],
    "evaluation_protocol": "评测流程、交叉验证、随机种子等公平性措施",
    "ground_truth": "gold standard 的来源和可靠性"
  }}}},
  "evaluation_threads": [
    {{{{
      "dimension_order": 1,
      "dimension_name": "评测维度名称（如 accuracy, scalability, robustness）",
      "main_findings": "该维度下的主要发现",
      "top_performers": ["表现最好的方法"],
      "bottom_performers": ["表现最差的方法"],
      "surprising_results": "反直觉的发现（如无则 null）",
      "confounding_factors": "可能影响结果的混杂因素",
      "evidence_location": "Figure X / Table Y"
    }}}}
  ],
  "figure_arguments": [
    {{{{
      "figure_id": "Figure 1 / Table 1",
      "argument_role": "performance_comparison|scalability_analysis|ablation_study|ranking_visualization|dataset_characteristics|method_overview|runtime_analysis|robustness_test",
      "compared_methods": ["该图比较了哪些方法"],
      "evaluation_dimension": "对应的评测维度",
      "key_takeaway": "该图的核心结论",
      "winner_context": "获胜方法在什么条件下获胜",
      "evidence_location": "page X"
    }}}}
  ],
  "rankings_and_recommendations": {{{{
    "overall_ranking": [
      {{{{
        "rank": 1,
        "method": "方法名",
        "strengths": "核心优势",
        "weaknesses": "核心劣势",
        "best_scenario": "最适用场景"
      }}}}
    ],
    "scenario_based_recommendations": [
      {{{{
        "scenario": "使用场景描述",
        "recommended_method": "推荐方法",
        "rationale": "推荐理由"
      }}}}
    ],
    "practical_guidelines": "作者给出的实践建议"
  }}}},
  "methodological_insights": {{{{
    "what_matters": "哪些因素对性能影响最大",
    "what_doesnt_matter": "哪些因素影响小于预期",
    "parameter_sensitivity": "参数敏感性发现",
    "computational_considerations": "计算资源/时间/内存需求对比"
  }}}},
  "limitations_of_benchmark": [
    {{{{
      "limitation": "评测本身的局限",
      "impact": "对结论可靠性的影响",
      "evidence_type": "stated|inferred"
    }}}}
  ],
  "literature_report": {{{{
    "topic_summary": "3-5段：评测问题背景和动机",
    "methods_overview": "4-6段：被评测方法的技术原理简介",
    "evaluation_narrative": "5-8段：评测结果按维度逐一叙述，含关键数据",
    "comparative_discussion": "3-5段：方法间差异的原因分析",
    "practical_implications": "2-3段：对实际使用者的指导意义",
    "benchmark_limitations": "2-3段：评测设计的局限性"
  }}}},
  "relevance_to_user": {{{{
    "pain_point_matches": [
      {{{{
        "user_pain_point": "来自用户画像的痛点",
        "paper_addresses": "评测中哪些结果直接回应该痛点",
        "actionability": "high|medium|low"
      }}}}
    ],
    "directly_applicable_methods": ["用户可直接采用的方法（基于 tech_stack 匹配）"],
    "datasets_reusable": ["用户可复用的数据集"],
    "parameter_recommendations": "基于评测结果对用户具体场景的参数建议"
  }}}},
  "research_recommendations": {{{{
    "method_selection_for_user": "结合用户数据特征和 pain_points，推荐最适合的方法",
    "pipeline_integration": "如何将推荐方法集成到用户的 tech_stack",
    "missing_evaluations": "评测没覆盖但对用户重要的维度",
    "reproduction_path": "复现评测：代码/数据/环境/参数"
  }}}}
}}}}

Return ONLY the JSON object."""


# ============================================================
# CALL 2 — DEEP ANALYSIS PASS (COMMENTARY)
# ============================================================

_DEEP_SYSTEM_ADDENDUM_COMMENTARY = """
This is the DEEP ANALYSIS PASS for a COMMENTARY/EDITORIAL/PERSPECTIVE paper.
The Foundation Pass results are provided as context.

## READING STRATEGY: Argument Reading
Your task is to DECONSTRUCT THE AUTHOR'S RHETORICAL ARGUMENT:
- What TRIGGERED this piece? (a recent paper, a controversy, a policy change, a trend)
- What is the author's CORE CLAIM? State it in one sentence.
- What EVIDENCE TYPE supports it? (empirical data, logical reasoning, appeal to authority, analogy)
- What COUNTERARGUMENTS does the author acknowledge? How convincingly are they addressed?
- What is the author's UNSTATED ASSUMPTION? (every argument rests on hidden premises)
- What is AT STAKE? If the community ignores this perspective, what could go wrong?
- How TIMELY is this? Why now and not 5 years ago or 5 years from now?

Commentary papers are OPINION pieces with varying degrees of evidentiary support.
Your job is to make the argument structure EXPLICIT so the reader can form their own judgment.
"""

_DEEP_DETAILED_MODE_COMMENTARY = """
DETAILED MODE (DEEP PASS — COMMENTARY): Analyze the complete argument structure, \
all supporting evidence, counterarguments, and implications. \
The following figure/table IDs were detected in the text (may be few or none):
  {figure_table_ids}
"""

_DEEP_USER_TEMPLATE_COMMENTARY = """## Paper Text:
{main_text}

## Retrieved Supplementary Material:
{supplement_text}

## Foundation Pass Results (use as context — do not repeat, only build on):
{foundation_json}

## User Research Context:
- Research field: {research_field}
- Research directions: {research_directions}
- Pain points: {pain_points}
- Tech stack: {tech_stack}
- Analysis focus: {focus}
- Goal: {goal}

{mode_instructions}

## Required JSON Schema (DEEP ANALYSIS PASS — COMMENTARY):
{{{{
  "commentary_structure": {{{{
    "trigger": "触发该评论的事件/论文/现象",
    "main_position": "作者的核心立场/论点",
    "argument_chain": [
      {{{{
        "step": 1,
        "claim": "论点",
        "evidence": "支撑证据或推理",
        "evidence_type": "empirical|logical|authority|analogy",
        "evidence_location": "page X / section Y"
      }}}}
    ],
    "counterarguments_acknowledged": [
      {{{{
        "counterargument": "作者承认的反对意见",
        "author_response": "作者如何回应",
        "evidence_location": "page/section"
      }}}}
    ],
    "call_to_action": "作者的呼吁或建议",
    "implications_if_correct": "如果作者正确，领域将如何发展"
  }}}},
  "figure_arguments": [
    {{{{
      "figure_id": "Figure 1 / Table 1 / Box 1",
      "argument_role": "position_illustration|data_supporting_argument|future_vision|comparison_of_views|conceptual_model",
      "content_description": "内容描述",
      "rhetorical_function": "在论证中的修辞功能",
      "evidence_location": "page X"
    }}}}
  ],
  "context_and_stakes": {{{{
    "field_context": "该评论发生在什么领域背景下",
    "what_is_at_stake": "如果忽视该观点，可能的后果",
    "timeliness": "该观点为什么现在提出而不是更早",
    "related_works": ["与该评论相关或被评论的原始论文"]
  }}}},
  "literature_report": {{{{
    "topic_summary": "2-3段：评论主题和背景",
    "argument_narrative": "3-5段：论证展开的逻辑叙述",
    "implications": "2-3段：影响和意义"
  }}}},
  "relevance_to_user": {{{{
    "pain_point_matches": [
      {{{{
        "user_pain_point": "来自用户画像的痛点",
        "paper_addresses": "该评论的观点如何与用户痛点相关",
        "actionability": "high|medium|low"
      }}}}
    ],
    "perspective_value": "该观点对用户研究视角的启发",
    "action_items": ["用户可能需要关注或响应的事项"]
  }}}},
  "research_recommendations": {{{{
    "position_assessment": "该观点的可靠性和局限性评估",
    "implications_for_user_work": "对用户在研方向的潜在影响",
    "follow_up_readings": ["建议跟进阅读的文献"]
  }}}}
}}}}

Return ONLY the JSON object."""


# ============================================================
# CALL 2 — DEEP ANALYSIS PASS (ATLAS / DATASET / RESOURCE)
# ============================================================

_DEEP_SYSTEM_ADDENDUM_ATLAS = """
This is the DEEP ANALYSIS PASS for a DATASET/ATLAS/RESOURCE paper.
The Foundation Pass results are provided as context.

## READING STRATEGY: Resource Evaluation Reading
Your task is to evaluate this paper AS A RESOURCE BUILDER, not as a hypothesis tester:
- SAMPLE DESIGN: How were samples selected? Are they representative? What biases exist in cohort/tissue/species selection?
- DATA QUALITY: What QC steps were applied? What was filtered out? Are the quality thresholds justified?
- ANNOTATION STRATEGY: How were cell types / features / entities annotated? Manual, automated, or hybrid? What's the inter-annotator agreement or validation?
- TECHNICAL PIPELINE: What sequencing/measurement platform? What preprocessing pipeline? Are batch effects controlled?
- RESOURCE VALUE: What can this dataset uniquely answer that no previous dataset could? What is the "moat"?
- REUSE POTENTIAL: How accessible is the data? What formats? What metadata is provided? Are there APIs or browsers?
- LIMITATIONS: What populations/conditions/tissues are MISSING? What questions CANNOT be answered with this resource?
- INITIAL FINDINGS: What biological insights did the authors extract as proof-of-concept? Are these compelling or superficial?

Dataset papers live or die by their DESIGN DECISIONS and ACCESSIBILITY.
Focus on whether the resource is well-designed, well-annotated, and genuinely reusable.
"""

_DEEP_DETAILED_MODE_ATLAS = """
DETAILED MODE (DEEP PASS — ATLAS/DATASET): Cover the complete resource design, all QC steps, \
annotation strategy, and initial analyses. Evaluate accessibility and reuse potential. \
The following figure/table IDs were detected in the text:
  {figure_table_ids}
"""

_DEEP_USER_TEMPLATE_ATLAS = """## Paper Text:
{main_text}

## Retrieved Supplementary Material:
{supplement_text}

## Foundation Pass Results (use as context — do not repeat, only build on):
{foundation_json}

## User Research Context:
- Research field: {research_field}
- Research directions: {research_directions}
- Pain points: {pain_points}
- Tech stack: {tech_stack}
- Analysis focus: {focus}
- Goal: {goal}

{mode_instructions}

## Required JSON Schema (DEEP ANALYSIS PASS — ATLAS/DATASET):
{{{{
  "resource_design": {{{{
    "resource_type": "atlas|reference_dataset|benchmark_dataset|database|tool_with_data",
    "organism": "物种",
    "tissue_or_system": "组织/系统/覆盖范围",
    "sample_design": {{{{
      "total_samples": "样本总数",
      "conditions_or_groups": ["条件/分组"],
      "selection_criteria": "样本选择标准",
      "representativeness": "代表性评估（覆盖了什么、遗漏了什么）",
      "potential_biases": "潜在偏倚（年龄、性别、种族、技术平台等）"
    }}}},
    "technology_platform": {{{{
      "sequencing_or_measurement": "测序/测量技术（如 10x Chromium, MERFISH, Visium）",
      "version_or_chemistry": "版本/化学试剂（如 v3, V2）",
      "resolution": "分辨率（单细胞/单核/空间分辨率）",
      "throughput": "通量（细胞数/样本数/基因数）"
    }}}},
    "preprocessing_pipeline": {{{{
      "alignment": "比对工具和参考基因组",
      "qc_filters": "质控过滤标准（基因数阈值、线粒体比例等）",
      "normalization": "标准化方法",
      "batch_correction": "批次效应校正方法（如有）",
      "integration_method": "数据整合方法（如有多批次/多平台）"
    }}}}
  }}}},
  "annotation_assessment": {{{{
    "annotation_strategy": "manual_expert|automated_transfer|hybrid|consensus",
    "cell_type_taxonomy": "使用的细胞类型分类体系",
    "annotation_depth": "注释层级深度（粗粒度 → 精细粒度）",
    "validation_method": "注释验证方法（marker基因、专家复审、跨数据集一致性）",
    "novel_types_discovered": ["新发现的细胞类型/状态"],
    "annotation_confidence": "注释置信度评估",
    "evidence_location": "page/section"
  }}}},
  "data_accessibility": {{{{
    "repositories": ["数据存储位置（GEO, CellXGene, HCA, Zenodo 等）"],
    "accession_numbers": ["登录号"],
    "data_formats": ["提供的数据格式（h5ad, loom, Seurat object 等）"],
    "metadata_richness": "元数据丰富程度评估",
    "interactive_browser": "是否提供交互式浏览器/门户",
    "api_access": "是否提供程序化接口",
    "code_availability": "分析代码的可获取性",
    "license": "数据使用许可"
  }}}},
  "figure_arguments": [
    {{{{
      "figure_id": "Figure 1 / Table 1",
      "argument_role": "resource_overview|qc_demonstration|annotation_validation|biological_discovery|comparison_with_existing|spatial_mapping|trajectory_analysis",
      "content_description": "图表内容描述",
      "demonstrates_what": "该图证明了资源的什么特性或发现",
      "data_subset": "使用了资源的哪个子集",
      "evidence_location": "page X"
    }}}}
  ],
  "initial_biological_findings": [
    {{{{
      "finding": "初步生物学发现",
      "novelty": "high|moderate|low",
      "requires_validation": true,
      "evidence_location": "Figure X / page Y",
      "methods_used": "使用的分析方法"
    }}}}
  ],
  "resource_comparison": {{{{
    "compared_to": ["与之对比的已有资源"],
    "advantages": ["本资源的优势"],
    "limitations": ["本资源的局限"],
    "complementary_resources": ["互补资源"]
  }}}},
  "limitations_and_gaps": [
    {{{{
      "limitation": "局限性描述",
      "category": "sample_coverage|technical|annotation|accessibility|temporal",
      "impact": "对资源使用的影响",
      "evidence_type": "stated|inferred"
    }}}}
  ],
  "literature_report": {{{{
    "topic_summary": "3-5段：资源构建的背景和动机",
    "resource_design_narrative": "4-6段：资源设计决策的详细叙述",
    "quality_assessment": "3-4段：数据质量和注释质量的评估",
    "biological_insights": "3-5段：初步发现的叙述",
    "comparison_and_positioning": "2-3段：与已有资源的对比定位",
    "reuse_guide": "2-3段：如何使用这个资源的实用指南"
  }}}},
  "relevance_to_user": {{{{
    "pain_point_matches": [
      {{{{
        "user_pain_point": "来自用户画像的痛点",
        "resource_addresses": "该资源如何帮助解决该痛点",
        "actionability": "high|medium|low"
      }}}}
    ],
    "reuse_scenarios": ["用户可以用该资源做什么（具体场景）"],
    "integration_with_user_data": "是否可以与用户现有数据整合，如何整合",
    "complementary_to_user_work": "该资源对用户在研项目的互补价值"
  }}}},
  "research_recommendations": {{{{
    "how_to_access": "获取和加载数据的具体步骤",
    "recommended_subset": "对用户最有价值的数据子集",
    "analysis_suggestions": "基于该资源可以做的分析（结合用户 pain_points）",
    "integration_strategy": "与用户数据整合的策略建议",
    "caveats_for_reuse": "使用时需要注意的陷阱和限制"
  }}}}
}}}}

Return ONLY the JSON object."""


# ============================================================
# HELPERS
# ============================================================

def _get_client(config: dict) -> OpenAI:
    return OpenAI(api_key=config["api_key"], base_url=config["base_url"])


def _build_supplement_text(retrieved_chunks: list[dict], max_chars: int = 12000) -> str:
    if not retrieved_chunks:
        return "(No supplementary materials provided or retrieved.)"
    parts = []
    total = 0
    for chunk in retrieved_chunks:
        meta = chunk["metadata"]
        header = (
            f"[{meta.get('source_file', '?')} | "
            f"Section: {meta.get('section', '?')} | "
            f"Relevance: {chunk.get('relevance_score', 0):.2f}]"
        )
        block = f"{header}\n{chunk['content']}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON. Returns dict with _parse_error on failure."""
    json_str = raw.strip()
    if json_str.startswith("```"):
        json_str = re.sub(r"^```(?:json)?\s*", "", json_str)
        json_str = re.sub(r"\s*```$", "", json_str.strip())
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to find JSON object boundaries
        start = json_str.find("{")
        end = json_str.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(json_str[start:end])
            except json.JSONDecodeError:
                pass
        return {"_raw_response": raw, "_parse_error": True}


def _call_llm_with_retry(
    client: OpenAI,
    config: dict,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    max_retries: int = 2,
    main_text_ref: list = None,   # mutable reference for truncation
    current_max_ref: list = None,  # [current_max_chars]
) -> dict:
    """
    Call LLM with retry logic. Returns parsed JSON or error dict.
    main_text_ref and current_max_ref allow caller to track truncation state.
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=config.get("temperature", 0.2),
            )
            raw = response.choices[0].message.content.strip()
            return _parse_json_response(raw)

        except Exception as e:
            last_error = str(e)
            err_lower = last_error.lower()

            if any(kw in err_lower for kw in [
                "data_inspection", "content_filter", "moderation", "inappropriate", "safety"
            ]):
                if current_max_ref and attempt < max_retries:
                    current_max_ref[0] = int(current_max_ref[0] * 0.6)
                    print(
                        f"  [Retry {attempt+1}/{max_retries}] Content filter — "
                        f"truncating to {current_max_ref[0]} chars...",
                        file=sys.stderr,
                    )
                    continue

            if any(kw in err_lower for kw in ["token", "context_length", "max_length", "too long"]):
                if current_max_ref and attempt < max_retries:
                    current_max_ref[0] = int(current_max_ref[0] * 0.5)
                    print(
                        f"  [Retry {attempt+1}/{max_retries}] Token limit — "
                        f"truncating to {current_max_ref[0]} chars...",
                        file=sys.stderr,
                    )
                    continue
            break

    return {"_api_error": last_error, "_parse_error": True}


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    text = text[:max_chars]
    last_break = text.rfind("\n\n")
    if last_break > max_chars * 0.8:
        text = text[:last_break]
    return text + "\n\n... [TRUNCATED]"


# ============================================================
# CALL 1 — FOUNDATION PASS
# ============================================================

def _call_foundation(
    main_text: str,
    supplement_text: str,
    profile: dict,
    llm_config: dict,
    mode: str,
    max_main_chars: int = 50000,
) -> dict:
    """Foundation pass: identity, reading cipher, key findings, concepts."""
    system_prompt = _SYSTEM_PROMPT + _FOUNDATION_SYSTEM_ADDENDUM

    mode_instr = _FOUNDATION_DETAILED_MODE if mode == "detailed" else _FOUNDATION_OVERVIEW_MODE

    current_max = [max_main_chars]
    client = _get_client(llm_config)

    for attempt in range(3):
        text = _truncate_text(main_text, current_max[0])

        user_prompt = _FOUNDATION_USER_TEMPLATE.format(
            main_text=text,
            supplement_text=supplement_text,
            research_field=profile.get("research_field", "academic research"),
            research_directions=", ".join(profile.get("research_directions", [])),
            pain_points=", ".join(profile.get("pain_points", [])),
            focus=profile.get("output_preference", {}).get("focus", "methods"),
            mode_instructions=mode_instr,
        )

        # Foundation pass is lighter: 6K overview, 8K detailed
        max_tokens = 8192 if mode == "detailed" else 6144

        result = _call_llm_with_retry(
            client, llm_config, system_prompt, user_prompt,
            max_tokens=max_tokens,
            max_retries=0,        # single attempt per outer loop iteration
            current_max_ref=current_max,
        )

        if not result.get("_parse_error"):
            return result

        # If parse error from LLM content issue, try reducing text
        if result.get("_api_error"):
            err = result["_api_error"].lower()
            if any(kw in err for kw in ["content_filter", "moderation", "safety"]):
                current_max[0] = int(current_max[0] * 0.6)
                continue
            if any(kw in err for kw in ["token", "context_length", "too long"]):
                current_max[0] = int(current_max[0] * 0.5)
                continue
        break

    return result


# ============================================================
# CALL 2 — DEEP ANALYSIS PASS
# ============================================================

def _call_deep(
    main_text: str,
    supplement_text: str,
    profile: dict,
    llm_config: dict,
    foundation: dict,
    figure_table_ids: Optional[list[str]] = None,
    max_main_chars: int = 50000,
    doc_type: str = "article",
) -> dict:
    """Deep pass: type-branched analysis (article/review/benchmark/commentary)."""
    # Select prompts based on document type
    _type_config = {
        "article": (_DEEP_SYSTEM_ADDENDUM, _DEEP_DETAILED_MODE, _DEEP_USER_TEMPLATE),
        "review": (_DEEP_SYSTEM_ADDENDUM_REVIEW, _DEEP_DETAILED_MODE_REVIEW, _DEEP_USER_TEMPLATE_REVIEW),
        "benchmark": (_DEEP_SYSTEM_ADDENDUM_BENCHMARK, _DEEP_DETAILED_MODE_BENCHMARK, _DEEP_USER_TEMPLATE_BENCHMARK),
        "commentary": (_DEEP_SYSTEM_ADDENDUM_COMMENTARY, _DEEP_DETAILED_MODE_COMMENTARY, _DEEP_USER_TEMPLATE_COMMENTARY),
        "atlas": (_DEEP_SYSTEM_ADDENDUM_ATLAS, _DEEP_DETAILED_MODE_ATLAS, _DEEP_USER_TEMPLATE_ATLAS),
    }
    sys_addendum, mode_template, user_template = _type_config.get(doc_type, _type_config["article"])

    system_prompt = _SYSTEM_PROMPT + sys_addendum

    ids_str = ", ".join(figure_table_ids) if figure_table_ids else "(none detected)"
    mode_instr = mode_template.format(figure_table_ids=ids_str)

    tech_stack_str = ", ".join(
        profile.get("tech_stack", {}).get("frameworks", [])
        + profile.get("tech_stack", {}).get("languages", [])
    )

    # Provide Foundation results as context (truncated to save tokens)
    foundation_summary = json.dumps({
        k: v for k, v in foundation.items()
        if k in ("title", "research_question", "paper_contribution_type",
                  "field_positioning", "key_findings", "reading_cipher", "concepts")
    }, ensure_ascii=False, indent=2)
    # Cap at 4000 chars to leave room for main text
    if len(foundation_summary) > 4000:
        foundation_summary = foundation_summary[:4000] + "\n... [TRUNCATED]"

    current_max = [max_main_chars]
    client = _get_client(llm_config)

    for attempt in range(3):
        text = _truncate_text(main_text, current_max[0])

        user_prompt = user_template.format(
            main_text=text,
            supplement_text=supplement_text,
            foundation_json=foundation_summary,
            research_field=profile.get("research_field", "academic research"),
            research_directions=", ".join(profile.get("research_directions", [])),
            pain_points=", ".join(profile.get("pain_points", [])),
            tech_stack=tech_stack_str,
            focus=profile.get("output_preference", {}).get("focus", "methods"),
            goal=profile.get("output_preference", {}).get("goal", "borrow_methods"),
            mode_instructions=mode_instr,
        )

        max_tokens = llm_config.get("max_tokens", 14336)
        if max_tokens < 12000:
            max_tokens = 14336

        result = _call_llm_with_retry(
            client, llm_config, system_prompt, user_prompt,
            max_tokens=max_tokens,
            max_retries=0,
            current_max_ref=current_max,
        )

        if not result.get("_parse_error"):
            return result

        if result.get("_api_error"):
            err = result["_api_error"].lower()
            if any(kw in err for kw in ["content_filter", "moderation", "safety"]):
                current_max[0] = int(current_max[0] * 0.6)
                continue
            if any(kw in err for kw in ["token", "context_length", "too long"]):
                current_max[0] = int(current_max[0] * 0.5)
                continue
        break

    return result


# ============================================================
# PUBLIC API
# ============================================================

def understand_paper(
    main_text: str,
    retrieved_chunks: list[dict],
    profile: dict,
    llm_config: dict,
    mode: str = "overview",
    figure_table_ids: Optional[list[str]] = None,
    max_main_chars: int = 50000,
    max_retries: int = 2,
    doc_type_override: Optional[str] = None,
) -> dict:
    """
    Orchestrate paper understanding.

    overview mode → Foundation Pass only  (single LLM call, ~6K tokens)
    detailed mode → Foundation + Deep Pass (two LLM calls, ~20K tokens total)

    Document type routing (v2.2):
    - Automatically detected from Foundation Pass paper_contribution_type
    - Can be overridden via doc_type_override ("article"|"review"|"benchmark"|"commentary")

    Returns merged dict with all fields. Deep pass errors are non-fatal —
    the Foundation pass result is always returned.
    """
    supplement_text = _build_supplement_text(retrieved_chunks)

    # --- Call 1: Foundation ---
    foundation = _call_foundation(
        main_text=main_text,
        supplement_text=supplement_text,
        profile=profile,
        llm_config=llm_config,
        mode=mode,
        max_main_chars=max_main_chars,
    )

    if foundation.get("_parse_error"):
        # Foundation failed — nothing to build on
        foundation["_foundation_failed"] = True
        return foundation

    foundation["_foundation_pass"] = True

    # --- Resolve document type for Deep Pass routing ---
    if doc_type_override:
        doc_type = doc_type_override
    else:
        doc_type = _resolve_deep_type(foundation)
    foundation["_doc_type"] = doc_type

    if mode != "detailed":
        # Overview mode: Foundation is sufficient
        return foundation

    # --- Call 2: Deep Analysis (type-branched) ---
    print(f"  Document type routing: {doc_type}", file=sys.stderr)
    deep = _call_deep(
        main_text=main_text,
        supplement_text=supplement_text,
        profile=profile,
        llm_config=llm_config,
        foundation=foundation,
        figure_table_ids=figure_table_ids,
        max_main_chars=max_main_chars,
        doc_type=doc_type,
    )

    if deep.get("_parse_error"):
        # Deep pass failed — return foundation with warning
        foundation["_deep_pass_error"] = deep.get("_api_error", "parse error")
        print(
            f"  [WARNING] Deep analysis pass failed: {foundation['_deep_pass_error']}. "
            "Foundation results are still available.",
            file=sys.stderr,
        )
        return foundation

    # Merge: Foundation fields take precedence for shared keys
    merged = {**deep, **foundation}
    merged["_foundation_pass"] = True
    merged["_deep_pass"] = True
    merged["_doc_type"] = doc_type
    return merged
