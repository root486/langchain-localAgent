# Agent-RAG 智能知识库检索系统

基于 LangChain ReAct Agent + RAG 的本地知识库智能检索工作台。Agent 自主编排检索策略，通过 Multi-Query 扩展、混合检索、交叉编码精排三级 pipeline 从本地知识库提取证据，结合 MCP 协议集成外部工具链，全链路 SSE 流式可观测。

- 对话、工具调用、检索过程全部可审计
- 长期记忆使用可直接编辑的 Markdown
- 技能不是黑盒函数，而是可读可改的 `SKILL.md`
- 前端直接展示流式回复、ThoughtChain 推理链和检索证据

---

## 项目特点

- **Agent 自主编排**：ReAct 推理-行动-观察循环，Agent 自主决策调用哪些工具、如何组合检索策略
- **MCP 协议集成**：通过 MCP stdio transport 以子进程方式接入 Tavily 联网搜索等外部工具，进程隔离、故障降级，新增 MCP Server 只需加一段配置
- **Multi-Query 查询扩展**：Router 自动判断查询类型（口语改写 / 子查询拆分），多路检索提升召回
- **混合检索 + 精排**：Vector + BM25 双路召回 → RRF 融合（宽池 Top-20）→ qwen3-rerank 交叉编码器精排（Top-6）
- **父子 Chunk + 上下文窗口**：句子边界切分，命中子 chunk 后向同 parent 兄弟扩展至 1500 字符，避免信息碎片化
- **自适应上下文压缩**：对话 token 接近窗口上限时自动触发滑动窗口压缩，近期消息保留原文，早期消息由 DeepSeek V4 Flash 蒸馏为摘要注入
- **Redis 可选**：会话热存储 + 用户偏好动态更新（不装也能正常跑，自动降级 JSON 文件）
- **Prompt 可解释**：系统提示词由 SOUL / IDENTITY / MEMORY / SKILLS 多个文件实时组装
- **技能可审计**：每个技能都是 `skills/*/SKILL.md`，Agent 按需读取执行

## 当前能力

- FastAPI + SSE 流式聊天
- ReAct Agent 工具编排（终端执行、Python REPL、文件读写、URL 抓取、MCP 外部工具）
- 会话持久化（Redis 热层 + JSON 文件冷层，7 天 TTL）
- 用户偏好动态更新（Redis Hash，Agent 对话中自动调整，替代静态 USER.md）
- 长期记忆向量检索（`backend/memory/MEMORY.md`）
- RAG 模式切换
- 本地知识库检索（Skill Agent → Multi-Query → Vector + BM25 → RRF → Rerank）
- 前端三栏工作台 + ThoughtChain 推理可视化
- 在线编辑 Memory / Skills / Workspace 文件

当前内置技能：

- `rag-skill`：本地知识库检索
- `web-search`：联网搜索（MCP Tavily）
- `get_weather`：天气查询（MCP Tavily）
- `retry-lesson-capture`：失败经验沉淀

## 检索链路

```mermaid
flowchart LR
    U["用户问题"] --> E["Multi-Query<br/>Router A/B"]
    E --> S["Skill Retriever Agent"]
    S --> V["向量检索"]
    S --> B["BM25 检索"]
    E --> V
    E --> B
    S --> F["RRF 融合<br/>宽池 Top-20"]
    V --> F
    B --> F
    F --> R["qwen3-rerank<br/>交叉编码器精排<br/>Top-6"]
    R --> G["LLM 生成回答"]
```

- **Multi-Query**（始终执行）：A 路由（口语/模糊）→ 主改写 + 同义改写 2 条；B 路由（对比/多部分）→ 子查询 ≤3 条
- **父子 Chunk**：句子边界切分，短 chunk（< 60 字符）自动合并，命中后上下文窗口扩展至 1500 字符
- **RRF 融合**（k=60）：Skill 证据 + 多路 Vector + 多路 BM25 汇总，取宽池 Top-20
- **qwen3-rerank 精排**：百炼交叉编码器对 Top-20 重排序，输出 Top-6，API 失败自动降级为原始排序

## 系统结构

```text
├─ backend/
│  ├─ api/                    # Chat、session、file、token、knowledge index 接口
│  ├─ cache/                  # Redis 客户端 + 用户偏好
│  ├─ graph/                  # Agent、prompt、session、memory 相关逻辑
│  ├─ knowledge/              # 仓库内置示例知识库
│  ├─ knowledge_retrieval/    # 检索链路：Skill → Multi-Query → Hybrid → RRF → Rerank
│  ├─ memory/                 # 长期记忆文件 MEMORY.md
│  ├─ scripts/                # 评测脚本（BM25 离线 + RAGAS 在线）
│  ├─ sessions/               # 会话历史（JSON + Redis）
│  ├─ skills/                 # 技能目录（SKILL.md 定义）
│  ├─ storage/                # 索引、manifest 等缓存
│  ├─ tools/                  # terminal / read_file / python_repl / fetch_url
│  ├─ workspace/              # SOUL.md / IDENTITY.md（系统提示词组件）
│  └─ app.py                  # FastAPI 入口 + MCP 初始化
├─ frontend/
│  ├─ src/app/                # 页面入口
│  ├─ src/components/         # UI 组件
│  └─ src/lib/                # API 客户端与状态管理
└─ README.md
```

## 技术栈

### 后端

- Python 3.10+
- FastAPI + Uvicorn
- LangChain 1.x（ReAct Agent + MCP Adapters）+ LlamaIndex（向量索引）
- MCP (Model Context Protocol) — stdio transport，`langchain-mcp-adapters` 桥接为 LangChain Tool
- Redis（可选，连接失败自动降级）
- OpenAI-compatible API（百炼 / 智谱 / DeepSeek / OpenAI）

### 前端

- Next.js 14 + React 18 + TypeScript
- Tailwind CSS + Monaco Editor

## 环境变量

示例文件见 [backend/.env.example](backend/.env.example)。最少配置：

```env
LLM_PROVIDER=zhipu
LLM_MODEL=glm-5
ZHIPU_API_KEY=your_key

EMBEDDING_PROVIDER=bailian
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_API_KEY=your_key

TAVILY_API_KEY=your_key
```

可选配置：

```env
# Redis（不配则纯文件模式）
REDIS_URL=redis://localhost:6379/0

# 摘要压缩用小模型
SUMMARY_MODEL=deepseek-v4-flash
SUMMARY_API_KEY=your_key
SUMMARY_BASE_URL=https://api.deepseek.com

# 模型上下文窗口
MAX_CONTEXT_TOKENS=128000
```

## 快速开始

### 1. 启动后端

```powershell
cd backend
conda activate D:\rag-env
python -m uvicorn app:app --host 127.0.0.1 --port 8004 --reload
```

健康检查：`http://127.0.0.1:8004/health`

启动日志中看到 `[MCP] 工具加载成功` 即表示 Tavily MCP 工具已就绪。

### 2. 启动前端

```powershell
cd frontend
cnpm run dev
```

默认地址：`http://localhost:3000`

### 3. 启动 Redis（可选）

```powershell
# 本地 Redis，不配也能正常用
redis-server.exe
```

## 评测

```bash
# BM25 离线评估（不需要 LLM，3 秒）
python backend/scripts/evaluate_faq_retrieval.py

# RAGAS 在线评测（faithfulness / relevancy / precision / recall / correctness）
python backend/scripts/evaluate_ragas.py --limit 10
python backend/scripts/evaluate_ragas.py --output storage/eval_outputs/ragas_full.json
```

## 常用接口

- `GET /health`
- `POST /api/chat`
- `GET /api/sessions`
- `GET /api/knowledge/index/status`
- `POST /api/knowledge/index/rebuild`

## 当前边界

- 适合本地开发和研究，不是生产级 SaaS
- 知识索引启动时首次构建较慢（需调用 embedding API），后续从磁盘加载
- Excel / PDF 依赖技能链路处理
