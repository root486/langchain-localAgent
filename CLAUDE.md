# CLAUDE.md — Mini-OpenClaw RAG 项目

> **改动代码前必读。** 本文档回答三个问题：
> 1. 项目长什么样（结构）
> 2. 消息怎么流转（数据流）
> 3. **改一个地方，还必须要一起改哪里（改动影响矩阵 ⭐）**
>
> 详细参考文档按需读取，路径见各节末尾。

---

## 一、项目概览

本地 RAG 智能体工作台：LangChain ReAct Agent + ChromaDB 向量检索 + FastAPI SSE 流式 + Next.js 三栏前端。

一句话数据流：`用户问题 → Agent 判断路径 → 检索证据注入上下文 → LLM 流式回答 → SSE 事件驱动前端渲染`。

**技术栈**
- 后端：Python 3.10+ / FastAPI / LangChain 1.x（create_agent）/ ChromaDB（嵌入式 SQLite，无需服务端）/ Redis（可选，自动降级）/ MCP（Tavily stdio + 高德地图 streamable http，见 `app.py::_init_mcp_tools`）/ LangSmith（检索链路追踪，`@traceable` 逐级包裹，见 4.8）
- 前端：Next.js 14 + React 18 + TypeScript + Tailwind + Monaco

**启动命令**
- 虚拟环境：后端依赖装在 conda 环境 `D:\rag-env`，先 `conda activate D:\rag-env`（或用 `D:\rag-env\python.exe` 直接调）
- 后端：`conda activate D:\rag-env && cd backend && python -m uvicorn app:app --host 127.0.0.1 --port 8004 --reload`（健康检查 `/health`）
- 前端：`cd frontend && cnpm run dev`（默认 3000 端口，API 默认指向 `http://127.0.0.1:8004/api`，见 `frontend/src/lib/api.ts` 的 `DEFAULT_API_PORT`）

---

## 二、目录结构

### 2.1 backend/（Python 后端）

| 路径 | 职责 | 关键依赖 |
|------|------|---------|
| `app.py` | FastAPI 入口：lifespan 初始化顺序、路由注册、MCP 工具加载（Tavily stdio + 高德 streamable http，带超时重试，见 `_init_mcp_tools` / `_load_server_tools`） | 所有单例、langchain_mcp_adapters |
| `api/chat.py` | `POST /api/chat` SSE 流、自动压缩触发（固定历史 token 预算 + 80% 硬兜底，后台化） | agent_manager、prompt_builder、config |
| `api/sessions.py` | 会话 CRUD + `generate-title` | session_manager |
| `api/files.py` | 读写白名单文件；**保存后副作用**（skills→snapshot） | agent_manager、skills_scanner |
| `api/tokens.py` | token 统计 | prompt_builder、session_manager |
| `api/knowledge_index.py` | 索引 status + rebuild | knowledge_indexer |
| `graph/agent.py` | **核心**：AgentManager.astream()，三条路径分支，SSE 事件生产 | memory_store、knowledge_orchestrator、session_manager、prompt_builder、tools |
| `graph/memory_store.py` | **长期记忆双端存储**：PostgreSQL `memories` 表（结构化事实源）+ ChromaDB `memory_facts` 召回；写入 PG+向量双写，读取向量召回→回 PG 取事实 | config、embeddings_client、chromadb、psycopg2 |
| `graph/prompt_builder.py` | 系统提示词组装（SOUL/IDENTITY/SKILLS + 长期记忆动态检索说明） | config |
| `graph/session_manager.py` | 会话持久化（Redis 热层 7 天 TTL + JSON 冷层） | redis_client |
| `knowledge_retrieval/types.py` | **数据契约**：Evidence / RetrievalStep / HybridRetrievalResult / OrchestratedRetrievalResult / IndexStatus + 枚举 | 全链路 + 前端 |
| `embeddings_client.py` | OpenAI 兼容 Embedding 客户端（httpx 直连，替代 LlamaIndex OpenAIEmbedding） | config |
| `knowledge_retrieval/indexer.py` | 知识库索引：md/json/pdf 切分、BM25、ChromaDB 向量（**Excel 不索引**） | config、types、embeddings_client、pymupdf |
| `knowledge_retrieval/hybrid_retriever.py` | 组合 vector + BM25 双路检索 | indexer、types |
| `knowledge_retrieval/orchestrator.py` | **检索编排**：Multi-Query → Hybrid → RRF → Rerank | query_expander、hybrid_retriever、fusion、reranker |
| `knowledge_retrieval/query_expander.py` | Multi-Query：Router A/B → 改写/分解查询 | config |
| `knowledge_retrieval/reranker.py` | qwen3-rerank 交叉编码器精排（百炼 API，失败降级） | config（用 embedding_api_key） |
| `knowledge_retrieval/fusion.py` | RRF 融合 | types |
| `cache/redis_client.py` | Redis 单例，连接失败自动降级 no-op | config |
| `cache/rag_cache.py` | 检索证据语义缓存：精确层（Redis/内存 dict）+ 语义层（内存余弦），命中跳过检索管线（`agent.py` 知识路径挂钩）；version=`last_built_at`，rebuild 自动失效 | config、redis_client、embeddings_client、knowledge_indexer |
| `tools/__init__.py` | `get_all_tools()` 组装普通对话 Agent 的工具 | fetch_url、python_repl、read_file、terminal |
| `tools/skills_scanner.py` | 扫描 `skills/*/SKILL.md` → `SKILLS_SNAPSHOT.md` | yaml |
| `scripts/` | 评测：`evaluate_faq_retrieval.py`（BM25 离线）、`evaluate_ragas.py`（两层：确定性检索 recall@k/MRR/nDCG 按阶段归因 + RAGAS 答案质量；`_ragas_compat.py` 为 ragas↔langchain-community 桥接补丁，须先于 import ragas） | |

### 2.2 源文件 / 生成物 / 用户数据

| 路径 | 性质 | 说明 |
|------|------|------|
| `backend/knowledge/` | 📄 源文件 | 知识库原始文件（md/json），**不可恢复，改后需手动重建索引** |
| `backend/memory/MEMORY.md` | 📄 ~~源文件~~ | **DEPRECATED**：长期记忆改为 PG `memories` 表 + ChromaDB `memory_facts`（见 graph/memory_store.py）；文件保留仅供查阅，已移出编辑白名单 |
| `backend/workspace/` | 📄 源文件 | 人格/身份（SOUL.md / IDENTITY.md），改后下次请求自动生效 |
| `backend/skills/*/SKILL.md` | 📄 源文件 | Skill 定义，增删改会自动重建 SKILLS_SNAPSHOT.md |
| `backend/SKILLS_SNAPSHOT.md` | 🔧 生成 | 启动 + 保存 skills 文件时自动重建 |
| `backend/storage/knowledge/manifest.json` | 🔧 生成 | chunk 数据 + BM25 tokens |
| `backend/storage/knowledge/vector/chroma/` | 🔧 生成 | ChromaDB 向量索引（chroma.sqlite3） |
| `backend/storage/memory_facts/` | 🔧 生成 | 长期记忆向量索引（ChromaDB `memory_facts/chroma/`，collection `memory_facts`） |
| `backend/sessions/*.json` | 💾 持久化 | 会话记录（Redis 热层 + 文件冷层） |
| `backend/sessions/archive/*.json` | 💾 持久化 | 自动压缩归档的消息 |

📄 = 源文件 | 🔧 = 生成物 | 💾 = 用户数据

### 2.3 frontend/（Next.js 前端）

| 路径 | 职责 |
|------|------|
| `src/lib/api.ts` | **类型定义 + API 客户端**：Evidence / RetrievalStep / KnowledgeIndexStatus / SessionHistory，与后端 `types.py` 一一对应 |
| `src/lib/store.tsx` | 全局状态 + **SSE onEvent 处理**：token / tool_start / tool_end / retrieval / new_response / done / title / error；`FIXED_FILES` 可编辑文件列表 |
| `src/components/chat/RetrievalCard.tsx` | 检索轨迹卡片，**STEP_META 按 stage 映射样式** |
| `src/components/chat/ThoughtChain.tsx` | 工具调用链展示（tool_start/tool_end 配对） |
| `src/components/chat/ChatMessage.tsx` / `ChatPanel.tsx` / `ChatInput.tsx` | 消息渲染 / 面板 / 输入 |
| `src/components/editor/InspectorPanel.tsx` | 右侧可编辑文件面板 |
| `src/components/layout/` | Sidebar / Navbar / ResizeHandle 布局 |

### 2.4 docs/

| 路径 | 内容 |
|------|------|
| `docs/contracts/SSE-protocol.md` | SSE 事件协议（**改事件必读**） |
| `docs/contracts/types-mapping.md` | 后端 dataclass ↔ 前端 type 映射 |
| `docs/internals/indexing-pipeline.md` | 索引构建数据契约（chunk dict 字段） |
| `docs/internals/vector-index-consistency.md` | 什么操作必须重建索引 |
| `docs/internals/storage-side-effects.md` | 文件保存副作用 / 配置变更影响表 |
| `docs/internals/init-order.md` | 初始化顺序与单例依赖 |
| `docs/internals/concurrency.md` | rebuild 并发与前端轮询边界 |
| `docs/checks/verification-checklist.md` | 修改后验证清单 |
| `docs/reference/hardcoded-values.md` | 关键硬编码值 |

---

## 三、核心数据流（三条路径）

`api/chat.py::chat` 接收用户消息 → `agent_manager.astream()` 按路径产生事件 → `chat.py` 包成 SSE 逐条转发 → 前端 `store.tsx onEvent` 渲染。

```
POST /api/chat (SSE)
  ▼ AgentManager.astream(message, history)
  │
  ├─ [路径1] 长期记忆检索（记忆常开；MEMORY.md 静态注入已废弃）
  │     memory_store.is_ready() 时 retrieve()（ChromaDB 召回 → 回 PG 取事实）
  │     → yield retrieval(kind=memory, stage=memory) → 证据注入 augmented_history
  │     → 继续走路径2 或 路径3
  │
  ├─ [路径2] 路由判断进知识库（agent.py `_route_knowledge_query`：显式关键词直通 + LLM 二分类，失败回退关键词，见 knowledge_retrieval/router.py）
  │     knowledge_orchestrator.astream(query):
  │       Multi-Query: expand() → 每条变体 hybrid_retriever.retrieve() → vector + bm25
  │       RRF 融合(宽池20) → rerank 精排(4) → yield orchestrated_result
  │     agent.py: 收到 orchestrated_result → 每个 step yield retrieval(kind=knowledge)
  │     → 证据注入 augmented_history → _astream_model_answer → token + done
  │
  └─ [路径3] 普通对话
        _build_agent() → agent.astream(stream_mode=["messages","updates"])
        → token 逐字 / tool_start / tool_end / new_response / done
```

**事件归属速查**（改事件前后端必须同步，见 4.2）：

| 事件 | 后端来源 | 前端处理 |
|------|---------|---------|
| `token` | agent.py | 拼 content |
| `tool_start` | agent.py | 追加 toolCall（output=""） |
| `tool_end` | agent.py | 填充最后一个 toolCall.output |
| `retrieval` | agent.py（memory）/ orchestrator（knowledge） | 追加 RetrievalStep |
| `new_response` | agent.py（普通对话） | 新建 assistant 消息 |
| `done` | agent.py | 兜底填充 content |
| `title` | chat.py（首条消息后） | 刷新会话列表 |
| `error` | chat.py 异常处理 | 显示错误 |
| `orchestrated_result` | 内部事件 | **前端不处理** |

---

## 四、改动影响矩阵 ⭐（改前必读）

> 核心规则：**改一个模块前，先查这张表，确认下游消费方，一并修改。**
> 原则：后端改 → 前端同步；契约层改 → 所有消费方同步；索引逻辑改 → 重建索引。

### 4.1 `types.py`（数据契约层）改动

`knowledge_retrieval/types.py` 是全链路 + 前端的数据契约中心，**任何字段改动都会波及下游**：

| 改动 | 必须同步的地方 |
|------|--------------|
| 改/增 `Evidence` 字段 | `indexer.py` / `fusion.py`（组装这个结构）→ `agent.py` 格式化 → **前端 `api.ts` 的 `Evidence` 类型** |
| 改/增 `RetrievalStep` 字段 | `orchestrator.py`（构造步骤）→ `agent.py`（`step.to_dict()` 成 retrieval 事件）→ **前端 `api.ts` `RetrievalStep` + `RetrievalCard.tsx` 渲染** |
| 改/增 `IndexStatus` 字段 | `indexer.py::status()` → `api/knowledge_index.py` → **前端 `api.ts` `KnowledgeIndexStatus` + `store.tsx` 轮询** |
| 新增 `RetrievalChannel` 值 | `Evidence.channel` 类型、`docs/contracts/types-mapping.md`、前端 `api.ts` 联合类型 |
| 新增 `RetrievalStep.stage` 值 | **前端 `RetrievalCard.tsx` 的 `STEP_META` 必须加样式**，否则 fallback 到默认样式 |

### 4.2 SSE 事件改动（后端 ↔ 前端）

| 改动 | 必须同步的地方 |
|------|--------------|
| 新增事件类型 | `api/chat.py` 转发（一般是自动的）+ **前端 `store.tsx` onEvent 必须加分支** + `docs/contracts/SSE-protocol.md` 事件表 |
| 改名/删除事件 | 前端 onEvent 同步改，否则消息丢失或成为死代码 |
| 改事件字段 | `store.tsx` 对应分支 + `api.ts` 类型（若透传到前端） |
| 新增 `RetrievalStep.stage` | 见 4.1 |
| 改 `retrieval` 事件 results 结构 | `store.tsx::normalizeEvidence` + `api.ts` `Evidence` |

> 事件顺序约束：`tool_start` 必须先于对应 `tool_end`（前端靠"最后一个 toolCall"配对）；`done` 必须是流中最后一个事件；`retrieval` 在 `token` 之前发出。

### 4.3 索引 / 存储改动（knowledge_retrieval/indexer.py / graph/memory_store.py）

| 改动 | 必须做的事 |
|------|-----------|
| 改 `_split_markdown` / `_split_json` / `_split_pdf` 输出字段 | **不能删改已有字段**（manifest.json + vector metadata + BM25 全部依赖）；改切分逻辑 → **删旧索引 + rebuild** |
| 改 `_build_bm25_index` / `retrieve_bm25` | BM25 用 **rank_bm25（BM25Okapi）+ jieba 分词**；索引在加载 manifest 时自动重建。**签名与 Evidence 输出格式必须保持不变**（hybrid_retriever / orchestrator / evaluate_ragas.py 依赖） |
| 改 `_tokenize`（分词）/ BM25 参数（k1,b） | **无需删索引**：tokens 不持久化，`_load_manifest` → `_build_bm25_index` 会用最新分词自动重算。BM25 参数见 `BM25_K1`（1.5）/ `BM25_B`（0.25，b 调低以适配长短不一的 FAQ 语料） |
| 改 embedding 模型 / provider | **删除 `storage/knowledge/vector/chroma/` + rebuild**，否则维度不匹配 |
| 增删 `knowledge/` 下文件 | 手动 `POST /api/knowledge/index/rebuild`（无自动检测） |
| 改 `rebuild_index()` | `finally` 块中 `_building = False` **不能删**（前端轮询依赖）；所有异常路径都要重置 |
| 改 `graph/memory_store.py` | PG `memories` 表与 ChromaDB `memory_facts` 必须双写一致（add 失败回滚）；检索阈值 `MIN_SCORE`、`top_k` 见 docs/reference/hardcoded-values.md |
| 改 `PG_DSN` / 换 PG 库 | 重启进程（lru_cache）；ChromaDB 向量侧在 `storage/memory_facts/`，与 PG 独立 |

### 4.4 配置改动（.env / config.py）

| 改动 | 必须做的事 |
|------|-----------|
| 改任何 `.env` 值 | **必须重启进程**（`get_settings()` 用 `@lru_cache(maxsize=1)`，进程内不刷新） |
| 改 `EMBEDDING_MODEL` / `EMBEDDING_PROVIDER` | 重启 + 删 vector 存储 + rebuild（见 4.3） |
| 改 `PG_DSN` / `DATABASE_URL` | 重启进程；改库后需在库内重建 `memories` 表（`memory_store.configure` 自动建） |
| 改 `RAG_ROUTER_ENABLED` / `RAG_ROUTER_MODEL` / `RAG_ROUTER_API_KEY` | **必须重启进程**；不配 Key 时路由模型回退 `SUMMARY_API_KEY` → 主模型（无需删索引） |
| 改 `AMAP_API_KEY` / `TAVILY_API_KEY` | **必须重启进程**；启动时按 key 存在与否加载对应 MCP server（`app.py::_init_mcp_tools`，缺失则跳过该工具，服务不受影响） |
| 新增配置项 | 同步改 `config.py` 的 `Settings` 解析 + `.env.example` + 读取该配置的模块 |
| 新增 provider | 同步改 `config.py` 的 `LLM_PROVIDER_DEFAULTS` / `EMBEDDING_PROVIDER_DEFAULTS` / `PROVIDER_ALIASES` + 各 `_resolve_*` 函数 |

### 4.5 文件白名单 / 可编辑文件

| 改动 | 必须同步的地方 |
|------|--------------|
| 新增可编辑文件类别 | **后端 `api/files.py` 的 `ALLOWED_PREFIXES` / `ALLOWED_ROOT_FILES` + 前端 `store.tsx` 的 `FIXED_FILES`**（两端白名单要一致） |
| 改 `api/files.py` 保存副作用 | 保存 `skills/*` → `refresh_snapshot()`；`knowledge/*` / `workspace/*` 无副作用。**`memory/` 已移出白名单**（长期记忆不再作为文件编辑） |

### 4.6 新增 Skill / 工具

| 改动 | 必须做的事 |
|------|-----------|
| 新增 `skills/*/SKILL.md` | 启动或保存后自动扫描进 `SKILLS_SNAPSHOT.md`；**前端 InspectorPanel 可编辑列表来自 `GET /api/skills`，无需手改**（但新增非 skills 白名单文件仍需按 4.5） |
| 新增 Agent 工具 | 注册到 `tools/__init__.py::get_all_tools()`（只有普通对话路径的 agent 拿到） |

### 4.7 其他模块依赖（改前确认）

| 改动 | 下游消费方 |
|------|-----------|
| `graph/agent.py::astream` 分支逻辑 | `api/chat.py`（SSE 包装）、前端（事件流） |
| `graph/agent.py::_route_knowledge_query` / `knowledge_retrieval/router.py` | 决定哪些消息走知识库路径（显式关键词 `STRONG_PATTERNS` 直通 + LLM 二分类 `route()`，失败回退 `FALLBACK_PATTERNS`；改任一 = 改路由行为） |
| `graph/prompt_builder.py::SYSTEM_COMPONENTS` | 新增组件 → 改元组 + 确保文件存在 |
| `graph/session_manager.py` 会话 JSON 格式 | `api/chat.py`（save_message）、`api/sessions.py`、`api/tokens.py`、前端 `api.ts` `SessionHistory` + `store.tsx::toUiMessages`、存量 `sessions/*.json` 与 archive |
| `knowledge_retrieval/orchestrator.py` 检索参数（hybrid top_k=4 / RRF 宽池 20 / rerank 精排 4） | **`scripts/evaluate_ragas.py` 模拟了同样的管线**，改参数要同步 |
| `graph/memory_store.py` 的 `add`/`retrieve` | `agent.py`（记忆检索分支、`_format_memory_retrieval_step`）、`graph/memory_store.py` 的 `MIN_SCORE`/`top_k`、PG `memories` 表结构；改 `retrieve` 返回字段需同步 `agent.py` 格式化 |
| `cache/rag_cache.py::get/put` | `agent.py` 知识路径（命中跳过 Multi-Query/混合检索/Rerank，仍生成答案）；改阈值/TTL/开关 → 同步 `.env` + 重启（4.4） |

### 4.8 LangSmith 追踪（smith.langchain.com）

检索链路追踪：`langsmith` + `@traceable` 逐级包裹（根 `rag_agent` = `agent.astream`），LangChain 模型调用（ChatOpenAI/LangGraph）自动上报。配置走 `LANGSMITH_*`（.env），无 Key 或 `LANGSMITH_TRACING=false` 时 `app.py` lifespan 强制关追踪（改 .env 需重启进程）。

| 改动 | 必须做的事 |
|------|-----------|
| 新增要追踪的检索/处理函数 | 用 `@traceable(run_type=..., name=...)` 包裹；**参数含 callable/model（如 `query_expander.expand` 的 `build_model`）必须加 `process_inputs` 剔除**，否则 orjson 序列化失败、run 永久 pending |
| 改 SSE 流 / 事件（见 4.2） | `done` 后根 run 才完成；客户端断开会触发 `GeneratorExit` 把根 run 标 error（chat.py 已把 title 生成移到流耗尽后规避，**别改回流内**） |
| 改 `.env` 的 `LANGSMITH_*` | **必须重启进程**（`get_settings()` 是 `@lru_cache`）；换项目改 `LANGSMITH_PROJECT`，历史 trace 留在旧项目 |
| 追踪函数 name / run_type | 仅影响 LangSmith UI 过滤与展示，不影响功能；run_type 枚举仅 tool/chain/llm/retriever/embedding/prompt/parser（无 reranker） |
| 评估脚本 | `evaluate_ragas.py` / `evaluate_faq_retrieval.py` 不走 lifespan，.env 有 Key 时评估会顺带上报；想避开 → 调用前 `LANGSMITH_TRACING=false` |

---

## 五、核心约束（硬规则）

### 5.1 索引不变性
- `_split_markdown` / `_split_json` / `_split_pdf` 输出的 dict 字段**不能删改**（doc_id / parent_id / source_path / source_type / locator / text / parent_text）
- BM25 用 **rank_bm25（BM25Okapi）+ jieba 分词**，索引在加载 manifest 时自动重建（`_build_bm25_index`），tokens 不持久化
- 改 embedding 模型 / 切分逻辑 → 必须删 `*/vector/chroma/` 旧索引再 rebuild；改分词 / BM25 参数无需删索引（加载时自动重算）
- 长期记忆改 embedding 模型 → 删 `storage/memory_facts/chroma/` + 清空 PG `memories` 的向量侧（`memory_store.add` 会重新嵌入）

### 5.2 前端兼容性
- 新增/改名/删除 SSE 事件 → 前端 `onEvent` 和类型定义必须同步更新
- 新增 `RetrievalStep.stage` / `RetrievalChannel` 枚举值 → 前端 `STEP_META` / 类型必须加样式
- 后端 dataclass 的 `to_dict()` 输出即前端 JSON 解析依据，**只能加字段，不能删**

### 5.3 并发与状态
- rebuild 期间不要改 retrieve 逻辑（可能读到半写状态）
- `rebuild_index()` 的 `finally` 块中 `_building = False` 不能删
- `memory_store.add`/`retrieve` 是同步调用（psycopg2 同步驱动），在 async astream 中会短暂阻塞事件循环；本地规模可接受，若变慢考虑换 asyncpg/线程池

### 5.4 初始化顺序
- 单例均为 import 时创建空实例，`app.py lifespan` 中按序 configure
- **不要在 configure 之前调用业务方法**（方法开头应有 `if self.base_dir is None: raise RuntimeError` 防御）

### 5.5 LangSmith 追踪
- 检索管线由 `@traceable` 逐级包裹成父子树：`memory_retrieve` / `route_knowledge_query` / `knowledge_orchestrator` → `multi_query_expand` → `hybrid_retrieve` → `vector_retrieve` + `bm25_retrieve` → `rrf_fusion` → `rerank`
- 参数含 callable 的追踪函数必须 `process_inputs` 剔除，否则 run 卡 pending
- `@traceable` 包 async generator：run 在生成器耗尽后才完成；客户端断开（SSE 收到 done 后关连接）会把根 run 标 `error(GeneratorExit)` —— chat.py 已把 title 生成移到流耗尽后规避，改 SSE 流时注意
- 无 `LANGSMITH_API_KEY` 或 `LANGSMITH_TRACING=false` 时 `app.py` lifespan 强制关追踪；`.env` 改动需重启进程

> 详细见：`docs/internals/init-order.md` · `docs/internals/concurrency.md`

---

## 六、初始化顺序（lifespan）

```
1. get_settings()                    # 加载 .env（@lru_cache）
2. 设置 LangSmith 环境变量           # 有 LANGSMITH_API_KEY → 开追踪；无 → 强制关（app.py lifespan 兜底）
3. refresh_snapshot()                # 生成 SKILLS_SNAPSHOT.md
4. _init_mcp_tools()                 # 按 key 加载 MCP 工具（Tavily stdio + 高德 streamable http；逐个 server 带超时+重试，单个失败不影响其它）
5. agent_manager.initialize(base_dir, mcp_tools)
     5a. SessionManager(base_dir)    # 创建 sessions 目录
     5b. get_all_tools(base_dir)     # 实例化 4 个工具
     5c. knowledge_orchestrator.configure()
6. memory_store.configure()          # PG 连接池 + 建 memories 表/索引 + ChromaDB memory_facts（失败自动降级）
7. knowledge_indexer.configure()     # load manifest + vector
8. if not ready: knowledge_indexer.rebuild_index()
```

依赖关系：`knowledge_orchestrator.configure` 在 `agent_manager.initialize` 内部调用，不能提前。

---

## 七、修改后必做验证

1. 后端启动正常、`GET /health` 返回 `ok`、`GET /api/knowledge/index/status` 返回 `ready: true`
2. 三条路径各测一遍：
   - 普通对话 → token 流 + done
   - 知识库查询（含"知识库"/"文档"/".pdf" 等触发词）→ retrieval + token + done
   - 长期记忆检索 → 相关提问时 retrieval(kind=memory) 出现；无关提问不注入（MIN_SCORE 阈值）
3. 前端渲染正常：RetrievalCard 显示检索步骤、ThoughtChain 显示工具调用
4. **修改检索逻辑后至少跑一次 BM25 离线评估**（走真实 `retrieve_bm25` 管线，rank_bm25 + jieba）：

```bash
python backend/scripts/evaluate_faq_retrieval.py
```

RAGAS 两层评测可用 `--no-generation --no-multi-query` 零 token 验证检索层（确定性，不走 LLM）：
```bash
python backend/scripts/evaluate_ragas.py --limit 5 --no-generation --no-multi-query
```

5. 改了 embedding/切分/分词 → 重建索引后确认 `vector_ready` 和 `bm25_ready` 都为 true
6. 改了追踪代码（@traceable / SSE 流）→ 发一条知识库查询，在 LangSmith `rag-project` 项目确认根 run `rag_agent` 与子节点全部 `success`（不应出现 `error(GeneratorExit)`）

> 完整 checklist：`docs/checks/verification-checklist.md` · 关键硬编码值：`docs/reference/hardcoded-values.md`
