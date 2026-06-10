---
name: hlokk-skill
description: >-
  学术文献个性化解读助手。解析PDF论文及附件，基于用户研究画像（YAML配置）
  精准提取信息，生成定制化研究建议。主文全文送LLM理解，补充材料通过embedding+RAG
  精准检索。支持任意学科领域，支持OpenAI兼容API，embedding和LLM独立配置。
  当用户提到论文解读、文献阅读、paper reading、文献分析、研究方法提取、
  解读paper、读论文、分析文章时使用此skill。
---

# Hlokk v2.2 — 学术文献全景解读

## 架构概览

五层流水线，overview 模式单次 LLM 调用，detailed 模式双次调用（Foundation + Deep）：

1. **文档接入层** (Python) — PDF解析、附件自动分类、section正则、图表ID全扫描
2. **文献理解层** (Python → 第三方LLM API) — 双次调用架构：
   - **Foundation Pass**（Call 1）：论文身份、贡献类型、领域定位、Reading Cipher（10字段）、核心发现、概念列表
   - **Deep Pass**（Call 2，仅 detailed 模式）：**按文献类型路由**到不同 JSON schema
3. **覆盖率检测层** (Python) — LLM 输出 vs 原文结构的覆盖缺口分析（类型化权重）
4. **个性化对齐层** (QoderWork, 按需触发) — 仅在用户明确要求对话中展示详细解读时执行；基于用户画像 + JSON 做相关性评分、阅读策略标注（POC/MOP/RPP/WIL）。默认跳过，以节省积分
5. **知识库写入层** (Python) — 可选 `--ingest` 生成 source card（Sources/）+ raw staging（raw/），供 hlokk-wiki Phase B 拆解为原子知识点

**积分策略**：重token的文献理解由第三方API承担；高判断的对齐和策略标注由QoderWork完成。

**多文献类型支持（v2.2）**：

采用三层架构：`Paper Classification → Reading Strategy Selection → Schema Selection → Deep Extraction`

| 文献类型 | 阅读策略 | Deep Pass 核心输出 |
|----------|---------|---------|
| **article** | Evidence Chain Reading（重构因果论证链） | research_threads, figure_arguments, core_methods |
| **review** | Landscape Reading（重建知识版图） | thematic_threads, consensus_and_debate, methods_landscape |
| **benchmark** | Evaluation Reading（评估评估者本身） | benchmark_design, evaluation_threads, rankings |
| **atlas** | Resource Evaluation Reading（评估资源设计决策） | resource_design, annotation_assessment, data_accessibility |
| **commentary** | Argument Reading（解构修辞论证） | commentary_structure, context_and_stakes |

自动检测映射：

| paper_contribution_type (Foundation 输出) | 路由类型 |
|---|---|
| `new_mechanism`, `new_method`, `new_framework`, `incremental_validation` | article |
| `review_synthesis` | review |
| `benchmark` | benchmark |
| `new_dataset`, `atlas`, `resource` | atlas |
| `commentary`, `editorial`, `perspective` | commentary |

**双调用 vs 单调用对比**：

| | overview（单调用） | detailed（双调用） |
|--|--|--|
| LLM调用次数 | 1 | 2 |
| 总输出token | ~6K | ~20K |
| Foundation字段 | ✓ | ✓ |
| Deep字段 | ✗ | ✓ |
| 适用场景 | 快速了解一篇文章 | 需要深度阅读/写入知识库 |

## 前置检查

工作目录: 用户选择的文件夹。

在运行前，检查以下文件是否存在：

- `config.yaml` — API配置（embedding + LLM分开）
- `profile.yaml` — 用户研究画像

如果缺失，从skill模板复制并提示用户填写：
```bash
cp ~/.qoderwork/skills/hlokk-skill/config_template.yaml <workspace>/config.yaml
cp ~/.qoderwork/skills/hlokk-skill/profile_template.yaml <workspace>/profile.yaml
```

Python依赖安装（首次）：
```bash
python -m pip install -r ~/.qoderwork/skills/hlokk-skill/scripts/requirements.txt
```

## 执行流程

### Step 1: 收集输入

从用户消息中提取：
- PDF文件路径（支持多个，包括附件）
- 本次关注点（可选，覆盖profile的session_focus）
- 是否有主文提示
- Zotero key（可选，用于知识库链接）
- 是否需要写入 Obsidian 知识库（`--ingest`）

### Step 2: 运行Python流水线

```bash
python ~/.qoderwork/skills/hlokk-skill/scripts/hlokk_main.py \
  --pdfs <pdf1> <pdf2> ... \
  --workspace <workspace_dir> \
  [--focus "本次关注点"] \
  [--main-hint "主文文件名关键词"] \
  [--skip-rag] \
  [--with-figures] \
  [--mode overview|detailed] \
  [--doc-type article|review|benchmark|commentary|atlas] \
  [--zotero-key <ZOTERO_KEY>] \
  [--ingest] \
  [--theme "Theme/SubTheme"]
```

**`--doc-type` 参数说明（v2.2 新增）：**
- 可选参数，用于手动覆盖文献类型自动检测
- 如不指定，自动从 Foundation Pass 的 `paper_contribution_type` 推断
- `article`（默认）— 原创研究论文，阅读策略：Evidence Chain Reading
- `review` — 综述/系统综述/meta-analysis，阅读策略：Landscape Reading
- `benchmark` — 方法评测/工具比较，阅读策略：Evaluation Reading
- `atlas` — 数据集/图谱/资源论文（如 Human Cell Atlas, Tabula Sapiens），阅读策略：Resource Evaluation Reading
- `commentary` — 短评/社论/观点文章，阅读策略：Argument Reading

**`--mode` 参数说明：**
- `overview`（默认）— 单次LLM调用，Foundation Pass，~6K tokens，覆盖核心发现+阅读密码
- `detailed` — 双次LLM调用，Foundation + Deep Pass，~20K tokens，全覆盖含图表论证+方法映射
- 未指定时根据 profile.yaml 中 `output_preference.depth` 自动推断（`deep_dive` → detailed，其余 → overview）

**`--ingest` + `--zotero-key` + `--theme` 参数（知识库写入 v4.0）：**
- `--ingest` — 流水线结束后自动调用 `hlokk_to_wiki.py`，生成两个文件：
  - **Source card**（薄卡片）→ `Sources/{short-name}.md`（`type: source`，一句话摘要 + 衍生知识点占位）
  - **Raw staging**（完整分析）→ `raw/{ZOTERO_KEY}_{short-name}.md`（`status: raw`，供 Phase B 拆解为 3-5 个原子知识点）
- `--zotero-key` — 提供 Zotero item key，写入 source card frontmatter，用于知识库链接
- `--theme` — 指定主题分类路径（如 `"Genomics/Data-Analysis"`），记录在 frontmatter 中

**Python 流水线内部步骤：**
1. 解析 PDF、分类主文/附件、提取 sections、扫描全部图表ID（文本引用，不渲染图片）
2. 切分附件文本为 chunks
3. Embedding + 向量存储（仅有附件时）
4. RAG 检索相关 supplement chunks（仅有附件时）
5a. **Foundation LLM Call**：论文身份、贡献类型、领域定位、reading_cipher、key_findings、concepts
5b. **Deep LLM Call**（仅 detailed 模式）：research_threads（叙事阶段排序）、figure_arguments（含论证角色/页码引用，不分析图片内容）、core_methods、literature_report、recommendations
6. ~~渲染图表页面图片（PyMuPDF）~~ 默认禁用，使用 `--with-figures` 开启
7. 覆盖率缺口检测
8. 生成交互式 HTML 报告
9. （可选）写入 HlokkObsidian：source card → Sources/ + raw staging → raw/

脚本完成后，在 `<workspace>/outputs/` 生成：
- **JSON** — 结构化理解结果
- **HTML** — 交互式可视化报告

如果启用 `--ingest`，额外在 vault 生成：
- **Source card** — `Sources/{short-name}.md`（薄卡片，`type: source`）
- **Raw staging** — `raw/{ZOTERO_KEY}_{short-name}.md`（完整分析，`status: raw`，待 hlokk-wiki Phase B 提取知识点）

### Step 3: 交付文件并报告运行状态

流水线完成后，将生成的文件链接交付给用户，并简要报告运行状态（成功/失败、覆盖率、模式等信息）。**不再在对话中输出详细解读报告**——所有分析内容已写入输出文件，用户可直接查看。

交付清单：
- **HTML 报告** — 交互式可视化报告
- **JSON 结构化数据** — 机读分析结果
- （如启用 `--ingest`）**Source card** + **Raw staging** — 知识库文件

## 输出交付格式（QoderWork 对话层）

流水线完成后，QoderWork 只在对话中输出**运行状态摘要**，不展开详细解读。所有分析内容已写入文件，用户通过文件链接查看。

```markdown
## Hlokk 运行完成

- **论文**: {title} ({journal_year})
- **模式**: {overview/detailed}
- **覆盖率**: {score}%（{状态}）
- **Deep Pass**: {完成/未完成/跳过}

**生成文件**：
- [HTML 交互报告](file:///path/to/report.html)
- [JSON 结构化数据](file:///path/to/output.json)
{--ingest 时额外显示：}
- Source card: `Sources/{short-name}.md`
- Raw staging: `raw/{ZOTERO_KEY}_{short-name}.md`

**简要提示**：
{如覆盖率 < 0.7，提示建议用 detailed 模式重跑}
{如 Deep Pass 失败，提示报告基于 Foundation 结果}
```

---

## 个性化对齐参考（按需使用）

当用户**明确要求**在对话中展示详细解读时，读取 JSON 按以下框架输出。默认情况下跳过此步骤。

**覆盖率预检**：
- ≥ 0.7 — 正常
- 0.5 ~ 0.7 — 提示建议 `--mode detailed`
- < 0.5 — 警告覆盖不足

**可选输出模块**（用户要求时按需组合）：相关性评分、研究建议、方法工具表、证据索引、叙事脉络树、阅读密码解码、图表论证分析、阅读策略标注（POC/MOP/RPP/WIL）。

详细 prompt 模板见 [reference.md](reference.md)。

## 输出格式

### 第一层：QoderWork 对话摘要（默认）

仅输出运行状态，见上文「输出交付格式」。

### 第二层：机读 JSON

已由Python流水线生成在 `outputs/` 目录，包含：
- Foundation字段：title, authors, journal_year, doi, paper_contribution_type, field_positioning, concepts, key_findings, reading_cipher, data_availability, code_availability
- Deep字段（detailed模式）：research_threads（含narrative_stage/stage_label）, figure_arguments（含argument_role/logical_necessity/weakness/weakness_type）, core_methods, section_summaries, limitations, tools_summary, statistical_methods, literature_report, relevance_to_user, research_recommendations
- 元数据：paper_id, analysis_mode, dual_call, zotero_key, coverage_gaps, detected_figure_table_ids

### 第三层：HlokkObsidian 写入（`--ingest` 时生成，v4.0）

由 `hlokk_to_wiki.py` 生成两个文件：

**Source card**（`Sources/{short-name}.md`）— 薄索引卡片：
- frontmatter: type=source, zotero_key, tags, themes
- 一句话摘要（来自 field_positioning 或 WTD）
- 衍生知识点占位（待 Phase B 填充 [[wikilink]]）

**Raw staging**（`raw/{ZOTERO_KEY}_{short-name}.md`）— 完整深度分析：
- frontmatter: status=raw, analysis_mode, themes, tags
- 核心问题与贡献、论文叙事脉络、核心发现、方法速查、图表论证、阅读密码速览、局限与开放问题
- hlokk-wiki 的 Phase B 会拾取此文件，拆解为 3-5 个原子知识点写入 `Knowledge/{theme}/`

## 特殊场景处理

**只有一个PDF（无附件）：**
使用 `--skip-rag` 跳过RAG流程。

**快速了解一篇文章：**
使用默认 `overview` 模式（单次LLM调用，~6K tokens）。

**需要深度阅读/写入知识库：**
使用 `--mode detailed`（双次LLM调用，~20K tokens）。

**阅读综述文献（v2.2）：**
```bash
--mode detailed --doc-type review
```
自动使用综述专用 Deep Pass 模板，提取主题线索、方法族图谱、领域共识与争议。如果 Foundation Pass 已正确识别为 `review_synthesis`，`--doc-type` 可省略。

**阅读方法评测/Benchmark 论文（v2.2）：**
```bash
--mode detailed --doc-type benchmark
```
自动使用评测专用模板，提取评测设计（方法/数据集/指标）、维度评测结果、方法排名和场景推荐。

**阅读评论/社论/观点文章（v2.2）：**
```bash
--mode detailed --doc-type commentary
```
使用 Argument Reading 策略，解构论证链、立场评估、领域影响。适合 1-5 页的短文。

**阅读数据集/图谱/资源论文（v2.2）：**
```bash
--mode detailed --doc-type atlas
```
使用 Resource Evaluation Reading 策略，重点评估样本设计、数据质量、注释策略、可重用性。适合 Human Cell Atlas、Tabula Sapiens、CellXGene 系列等大规模资源论文。

**写入知识库（v4.0）：**
```bash
--mode detailed --ingest --zotero-key <KEY> --theme "Genomics/Data-Analysis"
```
这会生成 source card → `Sources/` + raw staging → `raw/`。随后运行 hlokk-wiki Phase B 即可从 raw staging 中提取原子知识点。

**不知道应该放在哪个主题：**
省略 `--theme`，默认使用 `Computational-Methods/Foundation-Models`，之后可手动更新 frontmatter 的 `themes` 字段。

**覆盖率低 (< 0.7)：**
在个性化对齐报告中优先提示。overview 模式下通过 `--mode detailed` 提升覆盖率。

**Deep Pass 失败：**
JSON 中有 `_deep_pass_error` 字段，报告基于 Foundation 结果。可重新运行 `--mode detailed`。

**JSON解析失败：**
读取 `_raw_response` 字段，由QoderWork直接提取结构化信息。

## 附加资源

- 详细技术文档: [reference.md](reference.md)
- config模板: [config_template.yaml](config_template.yaml)
- profile模板: [profile_template.yaml](profile_template.yaml)
- Obsidian桥接脚本: [scripts/hlokk_to_wiki.py](scripts/hlokk_to_wiki.py)
