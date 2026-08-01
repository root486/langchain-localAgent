# 存储目录与隐式副作用

## 文件保存 API 的隐式触发规则

`POST /api/files` 保存文件后的副作用：

| 保存路径 | 触发的副作用 | 是否阻塞 |
|----------|-------------|---------|
| `skills/` 下的任何文件 | `refresh_snapshot()` → 重写 `SKILLS_SNAPSHOT.md` | 是 |
| `knowledge/` 下的任何文件 | **无** — 必须手动点"重建索引" | — |
| `workspace/` 下的文件 | **无** — 下次请求自动读取最新内容 | — |

**规则**：
- 如果要给 `knowledge/` 文件保存加自动 rebuild，需要异步执行（知识库可能很大），同时更新前端 UX 提示
- `memory/` 已移出编辑白名单（长期记忆改为 PG `memories` 表 + ChromaDB `memory_facts`，见 `graph/memory_store.py`），不再作为文件保存

## 配置变更影响表

`.env` 中的配置变更后，需要执行的操作：

| 变更项 | 需要重启? | 需要删除存储? | 需要重建索引? | 原因 |
|--------|----------|-------------|-------------|------|
| `LLM_*` 系列 | 是 | 否 | 否 | `get_settings()` 使用 `@lru_cache`，进程内不刷新 |
| `EMBEDDING_MODEL` | 是 | **是** (vector + memory_facts) | 是 | 向量维度可能变化；长期记忆侧需删 `storage/memory_facts/chroma/` 后由 `memory_store.add` 重新嵌入 |
| `EMBEDDING_PROVIDER` | 是 | **是** (vector + memory_facts) | 是 | 不同 provider 的模型不同 |
| `EMBEDDING_API_KEY` | 是 | 否 | 否 (启动时自动 rebuild) | 只影响能否生成向量 |
| `EMBEDDING_BASE_URL` | 是 | 否 | 否 | 只影响 API 端点 |
| `PG_DSN` | 是 | 否 | 否 | 改库后 `memory_store.configure` 会在目标库自动建 `memories` 表 |
| `TAVILY_API_KEY` | 是 | 否 | 否 | 只影响 web-search skill |
| `AMAP_API_KEY` | 是 | 否 | 否 | 启动时加载高德 MCP 工具（streamable http，`app.py::_init_mcp_tools`）；缺失则该工具不加载 |

**注意**：`get_settings()` 使用 `@lru_cache(maxsize=1)`，修改 `.env` 文件后**必须重启进程**才能生效。
