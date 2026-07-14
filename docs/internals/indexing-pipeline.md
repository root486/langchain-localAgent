# 后端内部数据契约 — 函数 I/O

## 索引构建链路

```
knowledge/ 目录文件
  │
  ▼ _split_markdown() / _split_json()
  │ 输出: list[dict] — 每个 dict 必须包含:
  │   doc_id, parent_id, source_path, source_type, locator, text, parent_text
  │
  ▼ _write_manifest()
  │ 输入: self._documents (上面的 list[dict])
  │ 输出: manifest.json = {"built_at": float, "documents": [...]}
  │
  ▼ _prepare_bm25_stats()
  │ 输入: self._documents — 每个 dict 必须有 "text" 字段
  │ 输出: 向每个 dict 注入 "tokens" 字段 + 更新 self._document_frequencies / self._avg_doc_length
  │ ⚠️ 副作用: 直接修改 self._documents 中的 dict，添加 "tokens" key
  │
  ▼ _build_vector_index()
    输入: self._documents — 每个 dict 的 metadata 需要: doc_id, parent_id, source_path, source_type, locator
    输出: VectorStoreIndex + 持久化到 storage/knowledge/vector/
```

**关键约束**：
- `_split_markdown` / `_split_json` 输出的 dict 字段是下游所有环节的依赖基础，**不能删除或改名已有字段**
- `_prepare_bm25_stats` 会向 dict 注入 `tokens` 字段，这个字段会被 `retrieve_bm25` 读取
- `_build_vector_index` 从 dict 中提取 5 个 metadata 字段构造 `Document`，缺少任何一个会导致 vector 检索结果中对应 metadata 为空

## 检索结果链路

```
retrieve_vector(query, top_k, path_filters)
  │ 输入: query: str, top_k: int, path_filters: list[str] | None
  │ 输出: list[Evidence] — channel 固定为 "vector"
  │ 依赖: self._vector_index (LlamaIndex VectorStoreIndex)
  │        metadata 中必须有 source_path (用于 path_filters 过滤)

retrieve_bm25(query, top_k, path_filters, query_hints)
  │ 输入: query: str, top_k: int, path_filters: list[str] | None, query_hints: list[str] | None
  │ 输出: list[Evidence] — channel 固定为 "bm25"
  │ 依赖: self._documents 中每个 dict 的 "tokens" 和 "text" 字段

HybridRetriever.retrieve(query, top_k, path_filters, query_hints)
  │ 输出: HybridRetrievalResult(vector_evidences, bm25_evidences)
  │ 纯组合，不做额外处理

reciprocal_rank_fusion(evidence_lists, top_k, rank_constant)
  │ 输入: Iterable[list[Evidence]]
  │ 输出: list[Evidence] — channel 改为 "fused"，score 改为 RRF 分数
  │ 去重 key: source_path|locator|snippet[:240] (归一化空白后)
```

## SkillRetrieverAgent LLM 输出解析

LLM 被要求输出 JSON，但实际输出不可控。解析链路：

```
LLM 原始文本
  ▼ _extract_json(text) → dict (可能为 {})
  ▼ payload.get("evidences", []) → 逐条解析为 Evidence
  ▼ payload.get("narrowed_paths", []) → narrowed_paths
  ▼ payload.get("narrowed_types", []) → _normalize_types()
  ▼ payload.get("rewritten_queries", []) → rewritten_queries
  ▼ payload.get("status", "") → 必须在 {success, partial, not_found, uncertain} 中
  ▼ 如果 status 不合法 → 根据 evidences 是否为空推断
```

**容忍度**：
- evidences 中缺少 source_path → 跳过该条（不崩溃）
- score 不是数字 → 设为 None
- status 不在枚举中 → 推断为 "success" 或 "uncertain"
- JSON 解析失败 → 返回空 dict，最终 status="uncertain"

**约束**：优化时不要降低这个容忍度，LLM 输出格式不稳定是常态。

## Evidence 字段在各渠道中的填充规则

| 字段 | vector | bm25 | skill | memory | fused |
|------|--------|------|-------|--------|-------|
| `source_path` | metadata.source_path | dict.source_path | LLM 输出 | "memory/MEMORY.md" | 继承原 Evidence |
| `source_type` | metadata.source_type | dict.source_type | LLM 输出 (默认 "unknown") | "memory" | 继承 |
| `locator` | metadata.locator | dict.locator | LLM 输出 | "memory" | 继承 |
| `snippet` | node.text | dict.text | LLM 输出 | node.text | 继承 |
| `channel` | "vector" | "bm25" | "skill" | "memory" | "fused" |
| `score` | float (similarity) | float (BM25) | float 或 None | float | float (RRF) |
| `parent_id` | metadata.parent_id | dict.parent_id | LLM 输出 | None | 继承 |
