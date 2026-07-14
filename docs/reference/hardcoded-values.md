# 关键硬编码值

| 值 | 位置 | 说明 |
|----|------|------|
| `chunk_size=1200` | `indexer.py` `_split_markdown` | Markdown 段落最大字符数 |
| `chunk_size=256, chunk_overlap=32` | `memory_indexer.py` | Memory 索引的 SentenceSplitter 参数 |
| `top_k=4` | `hybrid_retriever.py` | 向量/BM25 默认检索数量 |
| `top_k=3` | `memory_indexer.py` | Memory 默认检索数量 |
| `top_k=6` | `orchestrator.py` / `fusion.py` | RRF 融合后最终返回数量 |
| `rank_constant=60` | `fusion.py` | RRF 公式参数 |
| `k1=1.5, b=0.75` | `indexer.py` | BM25 评分参数 |
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
| `embed_batch_size=10` | `indexer.py` / `memory_indexer.py` | 百炼 embedding 批大小上限 |
| `AUTO_COMPRESS_RATIO=0.8` | `chat.py` | 自动压缩触发比例（模型窗口 80%） |
| `SESSION_REDIS_TTL=86400*7` | `session_manager.py` | Redis 会话过期时间（7 天） |

修改这些值时需要评估对下游的影响。
