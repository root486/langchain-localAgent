# 关键硬编码值

| 值 | 位置 | 说明 |
|----|------|------|
| `chunk_size=1200` | `indexer.py` `_split_markdown` | Markdown 段落最大字符数 |
| `top_k=3` | `graph/memory_store.py` | 长期记忆默认检索数量（agent.py 调用） |
| `MIN_SCORE=0.4` | `graph/memory_store.py` | 长期记忆相似度阈值，低于则不注入上下文（相关性门槛） |
| `top_k=4` | `hybrid_retriever.py` | 向量/BM25 默认检索数量 |
| `top_k=4` | `orchestrator.py`（rerank `top_n`）/ `fusion.py`（默认值） | RRF 精排后最终返回数量（宽池 20 → 精排 4，RAGAS 对比后采用 4） |
| `top_k=20` | `orchestrator.py` / `scripts/evaluate_ragas.py` | RRF 融合宽池（rerank 候选池），精排前保留的候选数量 |
| `rank_constant=60` | `fusion.py` | RRF 公式参数 |
| `k1=1.5, b=0.25` | `indexer.py` | BM25 评分参数（rank_bm25 BM25Okapi；b 调低以适配长短不一的 FAQ 语料，实测 b=0.25 命中率 95% > 旧实现 93.3%） |
| `similarity_top_k=top_k*4` | `indexer.py` | 向量检索过取倍数（用于 path_filter 后补足） |
| `terminal_timeout=30s` | `config.py` | 终端命令超时 |
| `python_repl_timeout=15s` | `python_repl_tool.py` | Python 代码执行超时 |
| `fetch_url_timeout=15s` | `fetch_url_tool.py` | URL 请求超时 |
| `read_file_max=10000` | `read_file_tool.py` | 文件读取最大字符数 |
| `terminal_max_output=5000` | `terminal_tool.py` | 终端输出截断字符数 |
| `component_char_limit=20000` | `config.py` | 系统提示词中每个组件的最大字符数 |
| `API_PORT=8004` | 前端 `api.ts` | 默认后端端口 |
| `MAX_CONTEXT_CHARS=1500` | `indexer.py` | 上下文窗口字符数上限 |
| `MIN_CHUNK_SIZE=60` | `indexer.py` | 短 chunk 合并阈值 |
| `EMBED_BATCH_SIZE=10` | `embeddings_client.py` | 百炼 embedding 单请求批大小上限（httpx 直连 OpenAI 兼容接口） |
| `AUTO_COMPRESS_RATIO=0.8` | `chat.py` | 自动压缩硬上限兜底比例（模型窗口 80%，只防溢出） |
| `AUTO_COMPRESS_TOKEN_LIMIT=12000` | `config.py` | 会话历史（含摘要链）token 预算，超过触发压缩（软触发） |
| `SUMMARY_CHAIN_TOKEN_LIMIT=3000` | `config.py` | 压缩摘要链 token 上限，超过二次折叠为单条 |
| `SESSION_REDIS_TTL=86400*7` | `session_manager.py` | Redis 会话过期时间（7 天） |
| `DEDUP_THRESHOLD=0.93` | `graph/memory_store.py` | 写入去重余弦阈值：与已有记忆相似度超过则视为重复跳过 |
| `MAX_MEMORIES=2000` | `graph/memory_store.py` | 记忆行数上限，超限删 created_at 最早的（锁死检索/去重扫描边界） |
| `RETRIEVE_POOL=3000` | `graph/memory_store.py` | retrieve 单次 query 的 n_results 上限（防御性，MAX_MEMORIES 下永不触达） |
| `EXTRACT_MESSAGES_MAX=20` | `graph/memory_store.py` | 抽取器：单次输入的最大消息条数 |
| `MEMORY_EXTRACT_MIN_MESSAGES=6` | `api/chat.py` | 记忆抽取触发阈值：距上次抽取的新消息 ≥6 条才后台抽取一次 |
| `memory_extracted_until` | `graph/session_manager.py` | 会话记录加性字段：记忆抽取游标（已抽取的消息条数），前端不读 |
| `RAG_CACHE_THRESHOLD=0.92` | `config.py` | 语义缓存命中余弦阈值（RAG 场景建议 0.90-0.95，调低会答非所问） |
| `RAG_CACHE_TTL=86400` | `config.py` | 检索证据缓存有效期（24h），Redis 精确层 TTL + 语义层惰性淘汰共用 |
| `RAG_CACHE_MAX_ENTRIES=500` | `config.py` | 内存语义索引最大条目数，超限淘汰最旧 |
| `ROUTER_TIMEOUT_SECONDS=8` | `knowledge_retrieval/router.py` | 知识库路由 LLM 判断超时（秒），超时按失败处理并回退关键词匹配 |
| `source_paths(max_entries=40)` | `indexer.py` | 路由 prompt 展示的知识库索引文件清单上限 |
| `MCP_LOAD_TIMEOUT=30s` | `app.py` `_load_server_tools` | 单个 MCP server 工具加载单次超时（`asyncio.wait_for`），超时按失败处理并重试 |
| `MCP_LOAD_RETRIES=3` | `app.py` `_load_server_tools` | MCP 工具加载重试次数，每次间隔 1s 退避；全部失败返回空列表，不影响服务 |
| `AMAP_HTTP_TIMEOUT=30s` | `app.py` amap server 配置 | 高德 MCP 单次 HTTP 请求超时（秒），API 挂时快速失败不挂死 |
| `AMAP_SSE_READ_TIMEOUT=60s` | `app.py` amap server 配置 | 高德 MCP SSE 读流超时（秒） |

修改这些值时需要评估对下游的影响。
