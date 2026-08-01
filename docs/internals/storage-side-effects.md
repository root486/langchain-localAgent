# 存储目录与隐式副作用

## 可编辑文件 API（已移除）

> 前端 Inspector 面板与后端 `api/files.py`（`GET/POST /api/files`、`GET /api/skills`）已删除，不再有"保存文件"入口。

- `SKILLS_SNAPSHOT.md` 现仅在**启动时**由 `tools/skills_scanner.refresh_snapshot()` 重建（app.py lifespan）；改 `skills/*/SKILL.md` 后需重启进程生效
- `workspace/`（SOUL.md / IDENTITY.md）：直接改文件，下次请求自动读取最新内容
- `knowledge/`：直接改文件后需手动 `POST /api/knowledge/index/rebuild`（无自动检测，见 `docs/internals/vector-index-consistency.md`）
- `memory/` 不是文件（长期记忆为 PG `memories` 表，见 `graph/memory_store.py`）

## 配置变更影响表

`.env` 中的配置变更后，需要执行的操作：

| 变更项 | 需要重启? | 需要删除存储? | 需要重建索引? | 原因 |
|--------|----------|-------------|-------------|------|
| `LLM_*` 系列 | 是 | 否 | 否 | `get_settings()` 使用 `@lru_cache`，进程内不刷新 |
| `EMBEDDING_MODEL` | 是 | **是** (vector + memories 表) | 是 | 向量维度可能变化；长期记忆侧需清空 PG `memories` 表后由 `memory_store.remember` 重新嵌入 |
| `EMBEDDING_PROVIDER` | 是 | **是** (vector + memories 表) | 是 | 不同 provider 的模型不同 |
| `EMBEDDING_API_KEY` | 是 | 否 | 否 (启动时自动 rebuild) | 只影响能否生成向量 |
| `EMBEDDING_BASE_URL` | 是 | 否 | 否 | 只影响 API 端点 |
| `PG_DSN` | 是 | 否 | 否 | 改库后 `memory_store.configure` 会在目标库自动建 `memories` 表 |
| `TAVILY_API_KEY` | 是 | 否 | 否 | 只影响 web-search skill |
| `AMAP_API_KEY` | 是 | 否 | 否 | 启动时加载高德 MCP 工具（streamable http，`app.py::_init_mcp_tools`）；缺失则该工具不加载 |

**注意**：`get_settings()` 使用 `@lru_cache(maxsize=1)`，修改 `.env` 文件后**必须重启进程**才能生效。
