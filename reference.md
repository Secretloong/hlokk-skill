# Hlokk 技术参考文档

## 目录结构

```
~/.qoderwork/skills/hlokk-skill/
├── SKILL.md                  # Skill 核心指令
├── reference.md              # 本文档
├── config_template.yaml      # API 配置模板
├── profile_template.yaml     # 用户画像模板
└── scripts/
    ├── requirements.txt      # Python 依赖
    ├── hlokk_main.py         # 主编排入口
    ├── pdf_parser.py          # PDF 解析 & 附件分类
    ├── text_chunker.py        # 语义分块
    ├── vector_store.py        # ChromaDB + Embedding
    ├── llm_client.py          # LLM API 调用
    └── utils.py               # 工具函数
```

用户工作目录（运行时）：
```
<workspace>/                   # 用户选择的工作目录
├── config.yaml               # 用户实际 API 配置
├── profile.yaml              # 用户实际研究画像
├── data/
│   └── chroma_db/            # 向量存储（自动生成）
├── outputs/                  # 生成的 JSON 报告
└── cache/                    # PDF 文本缓存（预留）
```

## API 配置详解

### Embedding API

| 平台 | base_url | 推荐模型 | 备注 |
|------|----------|----------|------|
| 火山引擎(豆包) | `https://ark.cn-beijing.volces.com/api/v3` | `doubao-embedding-large` | 需在火山方舟开通 |
| 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `text-embedding-v3` | DashScope OpenAI兼容模式 |
| OpenAI | `https://api.openai.com/v1` | `text-embedding-3-small` | 海外直连 |
| 硅基流动 | `https://api.siliconflow.cn/v1` | `BAAI/bge-large-zh-v1.5` | 国内性价比高 |

### LLM API

| 平台 | base_url | 推荐模型 | 备注 |
|------|----------|----------|------|
| 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` / `qwen-long` | qwen-long 适合长文 |
| 火山引擎(豆包) | `https://ark.cn-beijing.volces.com/api/v3` | `doubao-pro-32k` | 32K上下文 |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` | 性价比极高 |
| 硅基流动 | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-72B-Instruct` | 按量付费 |

### 配置示例：火山embedding + 百炼LLM

```yaml
embedding:
  api_key: "vol-xxxxxxxx"
  base_url: "https://ark.cn-beijing.volces.com/api/v3"
  model: "doubao-embedding-large"

llm:
  api_key: "sk-xxxxxxxx"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-plus"
  max_tokens: 8192
  temperature: 0.2
```

## 用户画像字段说明

| 字段 | 类型 | 作用 |
|------|------|------|
| `research_field` | string | 大领域，用于LLM上下文 |
| `research_directions` | list | 研究方向关键词，用于RAG query生成 |
| `tech_stack.languages` | list | 编程语言，用于判断复现可行性 |
| `tech_stack.frameworks` | list | 常用工具包，用于"是否已掌握"标注 |
| `tech_stack.environments` | list | 运行环境，用于复现路径建议 |
| `pain_points` | list | **最重要的字段**，直接驱动RAG检索query和相关性评分 |
| `output_preference.focus` | string | conclusions/methods/reproduction/figures |
| `output_preference.depth` | string | overview/standard/deep_dive |
| `output_preference.goal` | string | understand/reproduce/borrow_methods/write_paper |
| `session_focus` | string | 覆盖本次会话的关注点 |

## Python 流水线细节

### PDF 附件分类逻辑

基于文件名模式匹配 + 首页内容启发式：

1. 文件名含 `supplement/supp/SI/extended/appendix/additional` → supplement
2. 进一步细分: `methods` → supplementary_methods, `table` → supplementary_tables, etc.
3. 无匹配 → 按页数判断，最长文档为 main
4. 补充材料按优先级排序: methods > tables > figures > data > general

### 文本分块策略

- 优先按学术论文章节边界分块（Abstract, Methods, Results...）
- 章节内按段落边界滑动窗口，默认 512 tokens/chunk, 64 tokens overlap
- 超长段落回退到句子级分割
- 每个 chunk 保留元数据: source_file, section, page, chunk_id

### RAG 检索策略

Query 来源（按优先级）：
1. `session_focus`（如果指定）
2. `pain_points`（每个痛点一条 query）
3. `research_directions`
4. `pain_points × tech_stack.frameworks` 组合 query
5. 基于 `output_preference.focus` 的通用 query（如 focus=methods 时加 "parameters threshold pipeline"）

检索后去重、按余弦相似度排序，取 top_k（默认15）。

### LLM 理解层

- System prompt 约束: 证据定位、区分陈述/推断、精确工具参数、阅读密码提取、图表论证分析
- 输出: 严格 JSON schema，包含 core_methods, key_findings, tools_summary, statistical_methods, reading_cipher, figure_arguments, relevance_to_user
- 容错: JSON解析失败时保留原始响应，由QoderWork侧兜底

## 文献阅读汇报模块

### 设计原理

该模块将学术文献阅读方法论（阅读密码 + 阅读策略）系统化地嵌入 Hlokk 流水线。
分工原则：结构化提取（阅读密码、图表论证）由第三方 LLM API 承担；深度批判性分析（阅读策略）由 QoderWork 承担。

### 阅读密码（Reading Cipher）— 第三方 LLM 提取

| 密码 | 缩写 | 位置 | 含义 |
|------|------|------|------|
| 他们要做什么 | WTD | 前言 | 作者声称要在论文中做什么；提炼主要研究问题 |
| 现有文献综述 | SPL | 文献综述 | 前人研究结果的简要综述 |
| 现有文献批评 | CPL | 文献综述 | 现有文献的局限性和不足 |
| 研究空白 | GAP | 文献综述 | 现有文献中缺失的成分 |
| 理论依据 | RAT | 文献综述 | 基于GAP推导出研究的必要性 |
| 研究结果 | ROF | 结果/讨论 | 主要研究发现，通常在摘要/结果/结论反复强调 |
| 与文献一致 | RCL | 讨论 | 研究结果与现有文献观点一致的部分 |
| 与文献相反 | RTC | 讨论 | 研究结果与现有文献观点不一致的部分 |
| 他们做了什么 | WTDD | 结论 | 作者实际回答的主要研究问题和贡献 |
| 未来建议 | RFW | 结论 | 针对研究空白提出的未来研究方向 |

这些字段在 `llm_client.py` 的 JSON schema 中以 `reading_cipher` 对象输出。

### 图表论证分析 — 第三方 LLM 提取

每个主要 Figure/Table 生成一条 `figure_arguments` 记录：
- `figure_id`: 图/表编号
- `sub_argument`: 该图支撑的分论点
- `relation_to_theme`: 与核心主题的关系
- `data_source`: 使用的数据来源
- `methods_used`: 实验技术和分析方法
- `methods_assessment`: 方法的准确性、适用性或创新性评价

### 阅读策略（Reading Strategy）— QoderWork 生成

| 策略 | 缩写 | 含义 |
|------|------|------|
| 批评点 | POC | 现有文献中的缺陷，可供未来研究批评和弥补 |
| 明显遗漏点 | MOP | 作者忽视的与先前文献的理论/概念/方法的联系（常因文献阅读不充分） |
| 待探讨问题 | RPP | 未来研究中可进一步探讨的问题，可能成为新论文的切入点 |
| 能否 | WIL | 评估是否能通过逻辑梳理化解文章中的矛盾和待解决问题 |

阅读策略由 QoderWork 在个性化对齐层生成，因为需要结合用户画像做深度判断。
POC/MOP/RPP/WIL 的标注会与用户的 `pain_points` 和 `research_directions` 进行交叉匹配，
确保输出的批评和建议对用户当前研究有实际指导价值。

## CLI 完整参数

```
python hlokk_main.py \
  --pdfs <pdf1> [pdf2 ...] \     # 必需，PDF路径列表
  --workspace <dir> \             # 工作目录，默认当前目录
  --focus "关注点" \              # 覆盖 session_focus
  --main-hint "文件名关键词" \    # 强制指定主文
  --skip-rag                      # 跳过RAG（无附件或测试时）
```

## 未来扩展预留

### 跨文献对比（v2）
ChromaDB 向量库会持续积累，后续可实现：
- 同主题多篇文献的方法横向比较
- 基于历史文献库的相似论文检索
- 研究趋势追踪

### Ollama 本地模型支持（v2）
当前架构天然支持，只需在 config.yaml 中配置：
```yaml
llm:
  base_url: "http://localhost:11434/v1"
  model: "qwen2.5:14b"
  api_key: "ollama"     # Ollama 不校验 key，填任意值
```

### OCR 支持（v2）
扫描版PDF可通过 `pdf2image` + `pytesseract` 预处理，在 pdf_parser.py 中扩展。
