# CLAUDE.md — Mini-OpenClaw RAG 项目

> 本文档只保留架构概览 + 核心约束。详细参考文档按需读取，路径见各节末尾。

---

## 一、项目架构概览

### 数据流

```
用户消息 (前端 ChatInput)
  ▼ POST /api/chat (SSE stream)
  │
  ├─ Redis 会话热存储（读会话 → 降级 JSON 文件）
  ├─ Redis 用户偏好（动态更新，注入 system prompt）
  ├─ 自动压缩：token > 窗口 80% → DeepSeek Flash 摘要前一半消息
  │
  ▼ AgentManager.astream(message, history)
  ├─ rag_mode=true? → MemoryIndexer.retrieve() → 注入上下文
  ├─ 命中知识库关键词? → KnowledgeOrchestrator.astream()
  │     ├─ SkillRetrieverAgent (LLM Agent + ReadFile/Terminal/PythonRepl)
  │     ├─ Multi-Query: Router 判断 A/B → LLM 展开
  │     │     ├─ A（口语/模糊）→ 主改写 + 同义改写 2 条
  │     │     ├─ B（对比/多部分）→ 主改写 → 子查询分解 ≤3 条
  │     │     ├─ 每条变体 → HybridRetriever.retrieve()
  │     │     │     ├─ KnowledgeIndexer.retrieve_vector() → _build_context_window()
  │     │     │     └─ KnowledgeIndexer.retrieve_bm25() → _build_context_window()
  │     │     └─ 全部汇总 → RRF 融合（宽池 Top-20，含 Skill 证据）
  │     │     └─ qwen3-rerank 精排 → Top-6（百炼交叉编码器，¥0.5/百万 Token）
  │     └─ yields: retrieval steps, orchestrated_result
  └─ 普通对话 → LangChain Agent + 工具 → yields: token, tool_start, tool_end, done
  ▼ SSE events → 前端 store.tsx onEvent() → ChatMessage / RetrievalCard / ThoughtChain
```

### 目录地图

| 路径 | 性质 | 说明 |
|------|------|------|
| `backend/knowledge/` | 📄 源文件 | 知识库原始文件（md/json/txt），**不可恢复** |
| `backend/memory/MEMORY.md` | 📄 源文件 | 长期记忆原文 |
| `backend/workspace/` | 📄 源文件 | 人格文件（SOUL/IDENTITY），AGENTS.md 缺失 |
| `backend/skills/*/SKILL.md` | 📄 源文件 | Skill 定义 |
| `backend/cache/redis_client.py` | 🆕 新增 | Redis 单例，连接失败自动降级 |
| `backend/cache/user_prefs.py` | 🆕 新增 | Redis Hash 存储用户偏好，替代 USER.md |
| `backend/knowledge_retrieval/query_expander.py` | 🆕 新增 | Multi-Query 查询扩展（Router A/B） |
| `backend/knowledge_retrieval/reranker.py` | 🆕 新增 | qwen3-rerank 交叉编码器精排，RRF 融合后去噪 |
| `backend/SKILLS_SNAPSHOT.md` | 🔧 生成 | 下次启动自动重建 |
| `backend/storage/knowledge/manifest.json` | 🔧 生成 | chunk 数据 + BM25 tokens |
| `backend/storage/knowledge/vector/` | 🔧 生成 | LlamaIndex 向量索引文件 |
| `backend/storage/memory_index/` | 🔧 生成 | Memory 向量索引 |
| `backend/sessions/*.json` | 💾 持久化 | 会话记录（Redis 热层 + 文件冷层） |
| `backend/sessions/archive/*.json` | 💾 持久化 | 自动压缩归档的消息 |
| `backend/config.json` | 💾 持久化 | 运行时配置 |

📄 = 源文件 | 🔧 = 生成物 | 💾 = 用户数据 | 🆕 = 本次优化新增

---

## 二、核心约束

### 2.1 前端兼容性（改前必读）

- 新增/改名/删除 SSE 事件 → 前端 `onEvent` 和类型定义必须同步更新
- 新增 RetrievalChannel 枚举值 → 前端 `STEP_META` 必须加样式，否则 fallback 可能异常
- 新增 `SkillRetrievalResult.status` 值 → 必须考虑 `orchestrator.py` 的 fallback 逻辑（`partial/not_found/uncertain` 触发回退）

> 详细协议见: `docs/contracts/SSE-protocol.md` · `docs/contracts/types-mapping.md`

### 2.2 索引不变性（改前必读）

- `_split_markdown` / `_split_json` 输出的 dict 字段**不能删改**，下游全部依赖
- `_prepare_bm25_stats` 向每个 dict 注入 `tokens` 字段 → `retrieve_bm25` 依赖
- 改 embedding 模型/切分逻辑/分词逻辑 → **必须删旧索引再 rebuild**，否则 manifest 与 vector 不同步
- `rebuild_index()` 的 `finally` 块中 `_building = False` 不能删（前端轮询依赖）

> 详细约束见: `docs/internals/indexing-pipeline.md` · `docs/internals/vector-index-consistency.md`

### 2.3 并发与状态（改前必读）

- rebuild 期间不要改 retrieve 逻辑（可能读到半写状态）
- MemoryIndexer rebuild 是同步阻塞的，大文件注意超时
- 所有异常路径都要重置 `_building = False`

> 详细约束见: `docs/internals/concurrency.md` · `docs/internals/storage-side-effects.md`

### 2.4 初始化顺序

单例均为 import 时创建空实例，`app.py lifespan` 中按序 configure。**不要在 configure 之前调用业务方法。**

> 详细序列见: `docs/internals/init-order.md`

---

## 三、检索与上下文优化

### 3.1 父子 Chunk + 上下文窗口

- `_split_long_text()`：句子边界切分，不在句中硬切
- `_merge_short_chunks()`：同 parent 内过短 chunk（< 60 字符）合并
- `_build_context_window()`：命中子 chunk 后向同 parent 兄弟扩展，累计到 1500 字符
- Embedding monkey-patch：绕过 LlamaIndex 模型名校验，支持百炼 `text-embedding-v4`

### 3.2 Multi-Query 查询扩展

- `query_expander.py`：Router 判断 A/B → A（同义改写 2 条）/ B（子查询 ≤3 条）
- `orchestrator.py`：始终执行（不依赖 fallback），多路汇总 RRF 融合
- Skill Agent 的 `rewritten_queries` 作为 BM25 的 `query_hints` 混用

### 3.3 Rerank 精排

- `reranker.py`：调用百炼 `qwen3-rerank` 交叉编码器 API
- `POST https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank`
- RRF 融合后取宽池（Top-20），Reranker 精排到 Top-6
- 按输入 Token 计费（¥0.5/百万 Token），输出不计费，单次调用 ~5000 tokens
- API 失败自动降级为原始排序，不影响管线可用性
- 仅支持文本重排序，最多 500 文档/次，单条 ≤4000 tokens

### 3.4 Redis 集成

- `redis_client.py`：Redis 单例，连接失败自动降级
- `session_manager.py`：Redis 热层（7 天 TTL）+ JSON 文件冷层，读写先 Redis 后文件
- `user_prefs.py`：Redis Hash 存储用户偏好，替代 USER.md，Agent 对话中自动更新
- 已删除：`semantic_cache.py`（语义缓存，实验后移除）

### 3.5 自动上下文压缩

- 触发：全量 token（system + content + tool_calls + retrieval_steps）达模型窗口 80%
- 滑动窗口：`compress_history()` 保留后一半消息原文
- 摘要：DeepSeek V4 Flash 将前一半压成 200 字中文摘要，注入 `[以下是之前对话的摘要]`
- 旧手动压缩已删除（后端 API + 前端按钮 + store）

> 配置项：`.env` 中 `REDIS_URL`、`SUMMARY_MODEL/API_KEY/BASE_URL`、`MAX_CONTEXT_TOKENS`

---

## 四、修改后必做验证

1. 后端启动正常、`/health` 返回 ok、索引状态 `ready: true`
2. 普通对话和知识库查询 SSE 事件流完整（token → done）、前端渲染正常
3. 修改检索逻辑后至少跑一次 BM25 离线评估：

```bash
python backend/scripts/evaluate_faq_retrieval.py
```

> 完整 checklist 见: `docs/checks/verification-checklist.md` · 关键硬编码值见: `docs/reference/hardcoded-values.md`
