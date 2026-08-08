# 类型定义同步

## backend types.py ↔ frontend api.ts 映射

| 后端 dataclass | 前端 type | 关键同步字段 |
|---------------|-----------|-------------|
| `Evidence` | `Evidence` | `source_path, source_type, locator, snippet, channel, score, parent_id` |
| `RetrievalStep` | `RetrievalStep` | `kind, stage, title, message, results[]` |
| `HybridRetrievalResult` | — (不直接传前端) | — |
| `OrchestratedRetrievalResult` | — (拆成 retrieval 事件传前端) | — |
| `IndexStatus` | `KnowledgeIndexStatus` | `ready, building, last_built_at, indexed_files, vector_ready, bm25_ready` |

## 枚举值同步

**RetrievalChannel**（后端 `Literal` ↔ 前端联合类型）：
```
"memory" | "vector" | "bm25" | "fused"
```
- 后端加新值 → 前端 `api.ts` 的 `Evidence.channel` 类型必须同步更新
- 前端 `RetrievalCard.tsx` 的 `STEP_META` 必须加对应样式配置，否则 fallback 到默认样式

**memory 证据来源**（长期记忆为 ChromaDB 嵌入式 SQLite `storage/memory/chroma/`，见 `graph/memory_store.py`）：
- `source_path` = `memory`（恒值，不再区分 scope/category）
- `locator` = `memory`（恒值）

**RetrievalKind**：
```
"memory" | "knowledge"
```
- 前端 `normalizeRetrievalStep` 用 `item.kind === "memory"` 判断 → 改 kind 值会导致判断失败

**OrchestratedRetrievalResult.status**：
```
"success" | "not_found"
```
- 后端 `orchestrator.py` 在 RRF 融合产出证据时置 `success`，无证据时置 `not_found`
- 旧窄路径（skill_retriever_agent）的 `partial` / `uncertain` 已随其移除

## 修改后端 dataclass 的检查清单

- [ ] 后端 `to_dict()` 输出是否变化？→ 影响前端 JSON 解析
- [ ] 新增字段是否前端需要？→ 如果前端不显示可以不加，但不能删除已有字段
- [ ] 字段类型是否变化？→ 前端 `normalizeEvidence` 对 score 做了 string→number 兼容，但其他字段没有
- [ ] 枚举值是否变化？→ 前端类型定义和 STEP_META 必须同步
