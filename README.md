# Hlokk — AI-Powered Academic Paper Analysis

**[中文说明](#中文说明)** | **English**

A skill / tool for structured, personalized analysis of academic papers across **any discipline**. Hlokk parses PDFs, extracts sections with a three-layer hybrid strategy (font-aware + regex + LLM fallback), sends the full text to an LLM for structured understanding, and generates an interactive HTML report with research threads, figure arguments, reading cipher, and personalized recommendations.

Works with **QoderWork**, **OpenAI Codex**, **Claude Code**, **Cursor**, **Windsurf**, and any agent that can run shell commands.

## Features

- **Any academic discipline** — humanities, social sciences, STEM, medical, legal, etc.
- **Five-layer pipeline**: Document parsing → LLM understanding → Coverage detection → Personalization → Knowledge base writing
- **Multi-paper-type support**: Automatically routes article / review / benchmark / atlas / commentary to specialized extraction schemas
- **Dual LLM call architecture**: Foundation Pass (paper identity, reading cipher, key findings) + Deep Pass (research threads, figure arguments, methods, recommendations)
- **Three-layer section extraction**: Font-aware detection (pdfplumber) → Regex two-pass → LLM fallback with character offsets
- **RAG for supplementary materials**: Chunking → Embedding → ChromaDB vector store → Retrieval
- **Interactive HTML report**: Tabbed interface with overview, reading report, reading cipher, figure arguments, methods, personalized relevance, and evidence index
- **Obsidian knowledge base integration**: Optional `--ingest` to generate source cards and raw staging
- **OpenAI-compatible API**: Works with any OpenAI-compatible LLM provider

## Quick Install (AI Agent)

Hlokk is installed as a **skill** (or plugin) for your AI agent platform. Refer to your platform's official skill installation guide for the exact workflow:

- **QoderWork** — [Skill Installation Guide](https://docs.qoder.com/qoderwork/skills)
- **OpenAI Codex** — [Codex Agent Setup](https://platform.openai.com/docs/codex)
- **Claude Code** — [Claude Code Skills](https://docs.anthropic.com/en/docs/claude-code)
- **Cursor** — [Cursor Rules & Skills](https://docs.cursor.com/context/rules)
- **Windsurf** — [Windsurf Cascade](https://docs.windsurf.com/windsurf/cascade)

General steps:

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USER/hlokk-skill.git ~/.qoderwork/skills/hlokk-skill

# 2. Install Python dependencies
pip install -r ~/.qoderwork/skills/hlokk-skill/scripts/requirements.txt

# 3. Set up config
mkdir -p ~/hlokk-workspace
cp ~/.qoderwork/skills/hlokk-skill/config_template.yaml ~/hlokk-workspace/config.yaml
cp ~/.qoderwork/skills/hlokk-skill/profile_template.yaml ~/hlokk-workspace/profile.yaml

# 4. Edit config.yaml to add your API keys
# 5. Edit profile.yaml to set your research profile (see "Research Profile Best Practices" below)
```

## Platform Integration

### QoderWork

After installing to `~/.qoderwork/skills/hlokk-skill`, the skill is automatically available. Select your workspace folder and ask:

> "Read this paper: /path/to/paper.pdf"

### OpenAI Codex / Claude Code / Cursor / Windsurf

Add this to your agent's system prompt or project instructions:

```markdown
## Hlokk Paper Analysis

When asked to analyze a paper:
1. Run: python ~/.qoderwork/skills/hlokk-skill/scripts/hlokk_main.py --pdfs <path> --workspace <workspace> --mode detailed --skip-rag
2. Present the generated HTML report path to the user
3. If the user has supplementary materials, omit --skip-rag

Config files must exist in <workspace>: config.yaml (API keys) and profile.yaml (research profile).
```

Or use Hlokk as a library in your agent's code:

```python
import subprocess, json

result = subprocess.run([
    "python", "~/.qoderwork/skills/hlokk-skill/scripts/hlokk_main.py",
    "--pdfs", "/path/to/paper.pdf",
    "--workspace", "/path/to/workspace",
    "--mode", "detailed",
    "--skip-rag"
], capture_output=True, text=True)

# Read the generated JSON for structured data
with open("/path/to/workspace/outputs/paper.json") as f:
    analysis = json.load(f)
```

### CLI (standalone)

```bash
python ~/.qoderwork/skills/hlokk-skill/scripts/hlokk_main.py \
  --pdfs /path/to/paper.pdf \
  --workspace ~/hlokk-workspace \
  --mode detailed \
  --skip-rag
```

## Configuration

### config.yaml — API Keys

| Variable | Purpose | Example Providers |
|----------|---------|-------------------|
| `EMBEDDING_API_KEY` | Embedding service key | OpenAI, Cohere, Azure, DashScope |
| `EMBEDDING_BASE_URL` | Embedding service URL | `https://api.openai.com/v1` |
| `EMBEDDING_MODEL` | Model name | `text-embedding-3-small` |
| `HLOKK_LLM_KEY` | LLM service key | OpenAI, Anthropic (via proxy), Gemini |
| `HLOKK_LLM_URL` | LLM service URL | `https://api.openai.com/v1` |
| `HLOKK_LLM_MODEL` | Model name | `gpt-4o`, `claude-sonnet-4-20250514` |

Environment variables take priority over `config.yaml` values.

### profile.yaml — Research Profile

Describe your field, tools, pain points, and output preferences. This drives personalized relevance scoring and recommendations in the generated report.

**Research Profile Best Practices** — the more precise your profile, the better the personalized output:

1. **Best approach — AI-assisted distillation**: Ask your AI agent to read through your personal literature library (e.g., Zotero exports, published papers, reading notes) and distill a research profile from them. This produces the most accurate and grounded profile because it reflects what you actually read and work on.
2. **Good approach — self-description + AI refinement**: Write a draft describing your research interests, methods you use, and current challenges, then ask the AI to refine and expand it. This works well when your library is not yet digitized.
3. **Minimum viable**: Fill in the template with your best guess. The output will still be useful, but personalized recommendations will be less targeted.

## CLI Parameters

| Parameter | Description |
|-----------|-------------|
| `--pdfs` | PDF file paths (supports multiple) |
| `--workspace` | Directory containing config.yaml and profile.yaml |
| `--mode` | `overview` (1 LLM call, ~6K tokens) or `detailed` (2 calls, ~20K tokens) |
| `--main-hint` | Keyword to identify the main paper when multiple PDFs given |
| `--skip-rag` | Skip RAG pipeline (use when no supplementary materials) |
| `--with-figures` | Enable figure rendering (disabled by default) |
| `--doc-type` | Override: `article`, `review`, `benchmark`, `atlas`, `commentary` |
| `--ingest` | Generate Obsidian source card + raw staging |
| `--zotero-key` | Zotero item key for knowledge base linking |
| `--theme` | Obsidian vault theme path |
| `--focus` | Override session focus from profile.yaml |

## Output

After running, files are generated in `<workspace>/outputs/`:

- **HTML report** — Interactive tabbed report
- **JSON** — Structured machine-readable analysis

With `--ingest`, additionally in your Obsidian vault:

- **Source card** — `Sources/{short-name}.md`
- **Raw staging** — `raw/{ZOTERO_KEY}_{short-name}.md`

> **Language note**: The generated reports are primarily in **Chinese (中文)**. If you need output in other languages, ask your AI agent to modify the skill's system prompt and template strings to adapt to your preferred language.
>
> **JSON output**: The structured JSON contains all extracted knowledge points (paper identity, reading cipher, research threads, figure arguments, methods, recommendations, etc.) and can be directly integrated into your personal knowledge base system.

## Architecture

```
PDF(s) → [Document Parser] → [Text Chunker] → [Embedding + ChromaDB]
              ↓                                      ↓
         main text + sections              RAG-relevant supplement chunks
              ↓                                      ↓
         ┌─────────────────────────────────────────────┐
         │  LLM Foundation Pass (Call 1)               │
         │  → paper identity, reading cipher,           │
         │    key findings, concepts                    │
         ├─────────────────────────────────────────────┤
         │  LLM Deep Pass (Call 2, detailed only)      │
         │  → research threads, figure arguments,       │
         │    core methods, literature report,          │
         │    recommendations                           │
         └─────────────────────────────────────────────┘
              ↓
         [Coverage Detector] → [HTML Report Generator]
              ↓ (optional)
         [Obsidian Wiki Writer] → Source card + Raw staging
```

## Supported Paper Types

| Type | Reading Strategy | Deep Pass Output |
|------|-----------------|------------------|
| **article** | Evidence Chain Reading | research_threads, figure_arguments, core_methods |
| **review** | Landscape Reading | thematic_threads, consensus_and_debate, methods_landscape |
| **benchmark** | Evaluation Reading | benchmark_design, evaluation_threads, rankings |
| **atlas** | Resource Evaluation Reading | resource_design, annotation_assessment, data_accessibility |
| **commentary** | Argument Reading | commentary_structure, context_and_stakes |

## License

MIT

---

## 中文说明

**Hlokk** — 面向**任意学科**的 AI 学术文献结构化解读工具。可作为 QoderWork 技能、CLI 工具、或嵌入到 Codex / Claude / Cursor 等 AI 编码平台中使用。

### 核心能力

- **全学科覆盖** — 自然科学、社会科学、工程、医学、人文等均可
- **五层流水线**：文档解析 → LLM 理解 → 覆盖率检测 → 个性化对齐 → 知识库写入
- **多文献类型支持**：自动路由 article / review / benchmark / atlas / commentary 到专用提取 schema
- **双 LLM 调用架构**：Foundation Pass（论文身份、阅读密码、核心发现）+ Deep Pass（研究线索、图表论证、方法映射、文献汇报、研究建议）
- **三层 section 提取**：字体感知检测 → 正则两轮匹配 → LLM 回退（字符偏移定位）
- **补充材料 RAG**：分句切块 → Embedding → ChromaDB → 检索
- **交互式 HTML 报告**：七个标签页覆盖概览、文献汇报、阅读密码、图表论证、方法工具、个性化解读、证据索引
- **Obsidian 知识库桥接**：可选 `--ingest` 生成 source card 和 raw staging
- **OpenAI 兼容 API**：支持 OpenAI、DashScope、火山引擎、Ollama 等任意兼容接口

### 快速安装

Hlokk 以 **skill（技能/插件）** 的形式安装到你的 AI 代理平台中。请参考所属平台的官方 skill 安装指引完成安装：

- **QoderWork** — [技能安装指南](https://docs.qoder.com/qoderwork/skills)
- **OpenAI Codex** — [Codex Agent Setup](https://platform.openai.com/docs/codex)
- **Claude Code** — [Claude Code Skills](https://docs.anthropic.com/en/docs/claude-code)
- **Cursor** — [Cursor Rules & Skills](https://docs.cursor.com/context/rules)
- **Windsurf** — [Windsurf Cascade](https://docs.windsurf.com/windsurf/cascade)

通用安装步骤：

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USER/hlokk-skill.git ~/.qoderwork/skills/hlokk-skill

# 2. 安装依赖
pip install -r ~/.qoderwork/skills/hlokk-skill/scripts/requirements.txt

# 3. 初始化工作目录
mkdir -p ~/hlokk-workspace && \
  cp ~/.qoderwork/skills/hlokk-skill/config_template.yaml ~/hlokk-workspace/config.yaml && \
  cp ~/.qoderwork/skills/hlokk-skill/profile_template.yaml ~/hlokk-workspace/profile.yaml

# 4. 编辑 config.yaml，填入你的 API key
# 5. 编辑 profile.yaml，设置你的研究画像（参见下方"研究画像构建建议"）
```

### 多平台适配

**QoderWork**：安装后自动可用，选择工作目录后对话即可。

**Codex / Claude Code / Cursor / Windsurf**：在系统提示词或项目规则中添加：

```
分析论文时运行: python ~/.qoderwork/skills/hlokk-skill/scripts/hlokk_main.py --pdfs <路径> --workspace <工作目录> --mode detailed --skip-rag
```

**CLI 独立使用**：

```bash
python ~/.qoderwork/skills/hlokk-skill/scripts/hlokk_main.py \
  --pdfs /path/to/paper.pdf \
  --workspace ~/hlokk-workspace \
  --mode detailed \
  --skip-rag
```

> **输出语言**：生成的解读报告以**中文**为主。如需其他语言，可让 AI 代理直接修改 skill 的系统提示和模板字符串来适配目标语言。
>
> **JSON 输出**：结构化 JSON 包含所有提取的知识点（论文身份、阅读密码、研究线索、图表论证、方法工具、研究建议等），可直接整合入个人知识库构建系统。

### 配置说明

**config.yaml** — 配置 embedding 和 LLM 服务的 API key、base_url、model。支持环境变量覆盖。

**profile.yaml** — 描述你的研究领域、技术栈、当前痛点和输出偏好，驱动报告中的个性化相关性评分。

**研究画像构建建议** — 画像越精准，个性化输出质量越高：

1. **最佳方式 — AI 蒸馏**：让 AI 阅读你的个人文献库（如 Zotero 导出、已发表论文、阅读笔记），从中蒸馏出研究画像。这种方式最准确，因为它反映的是你真实的研究兴趣和方向。
2. **次选方式 — 自行描述 + AI 补充**：先自己写一份研究兴趣、常用方法、当前挑战的草稿，再让 AI 帮你润色和扩展。适合文献库尚未数字化的情况。
3. **最低门槛**：直接在模板中填入你的大致方向。输出仍然有用，但个性化建议的针对性会弱一些。

### 许可证

MIT
