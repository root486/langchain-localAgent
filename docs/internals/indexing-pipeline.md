# 后端内部数据契约 — 函数 I/O

## 索引构建链路

```
knowledge/ 目录文件
  │
  ▼ _split_markdown() / _split_json() / _split_pdf()
  │   md：按标题层级切分；json：按记录切分；pdf：PyMuPDF 按页提取文本、页内按句子边界切分
  │ 输出: list[dict] — 每个 dict 必须包含:
  │   doc_id, parent_id, source_path, source_type, locator, text, parent_text
  │ 注: .xlsx/.xls 明确不进索引（表格行数据对 RAG 召回价值低）
  │
  ▼ _write_manifest()
  │ 输入: self._documents (上面的 list[dict])
  │ 输出: manifest.json = {"built_at": float, "documents": [...]}
  │
  ▼ _build_bm25_index()
  │ 输入: self._documents — 每个 dict 必须有 "text" 字段
  │ 输出: 用 rank_bm25 构建 BM25Okapi 索引（k1=1.5, b=0.25），保存到 self._bm25
  │ 分词: jieba（中文）+ 字母数字正则（英文/数字），见 _tokenize
  │ ⚠️ tokens 不持久化，每次加载 manifest 时用最新分词重建 BM25 索引
  │
  ▼ _build_vector_index()
    输入: self._documents — 每个 dict 的 metadata 需要: doc_id, parent_id, source_path, source_type, locator
    输出: ChromaDB collection "knowledge_chunks"（cosine 空间）+ 持久化到 storage/knowledge/vector/chroma/
```

**关键约束**：
- `_split_markdown` / `_split_json` / `_split_pdf` 输出的 dict 字段是下游所有环节的依赖基础，**不能删除或改名已有字段**
- BM25 索引（`BM25Okapi`）由 `_build_bm25_index` 在加载 manifest 时构建，**不向 dict 注入任何字段**，改分词/BM25 参数无需重建索引
- `_build_vector_index` 从 dict 中提取 5 个 metadata 字段写入 ChromaDB metadata，缺少任何一个会导致 vector 检索结果中对应 metadata 为空

## 检索结果链路

```
retrieve_vector(query, top_k, path_filters)
  │ 输入: query: str, top_k: int, path_filters: list[str] | None
  │ 输出: list[Evidence] — channel 固定为 "vector"
  │ 依赖: self._vector_collection (ChromaDB Collection；cosine 距离转相似度 score = 1 - distance)
  │        metadata 中必须有 source_path (用于 path_filters 过滤)

retrieve_bm25(query, top_k, path_filters, query_hints)
  │ 输入: query: str, top_k: int, path_filters: list[str] | None, query_hints: list[str] | None
  │ 输出: list[Evidence] — channel 固定为 "bm25"
  │ 依赖: self._bm25 (BM25Okapi) + self._documents 中每个 dict 的 "text" 字段

HybridRetriever.retrieve(query, top_k, path_filters, query_hints)
  │ 输出: HybridRetrievalResult(vector_evidences, bm25_evidences)
  │ 纯组合，不做额外处理

reciprocal_rank_fusion(evidence_lists, top_k, rank_constant)
  │ 输入: Iterable[list[Evidence]]
  │ 输出: list[Evidence] — channel 改为 "fused"，score 改为 RRF 分数
  │ 去重 key: source_path|locator|snippet[:240] (归一化空白后)
```

## Evidence 字段在各渠道中的填充规则

| 字段 | vector | bm25 | memory | fused |
|------|--------|------|--------|-------|
| `source_path` | metadata.source_path | dict.source_path | "memory:{scope}/{category}" | 继承原 Evidence |
| `source_type` | metadata.source_type | dict.source_type | "memory" | 继承 |
| `locator` | metadata.locator | dict.locator | "memory" | 继承 |
| `snippet` | `_build_context_window(hit_doc)`（manifest 兄弟 chunk 扩展） | dict.text | ChromaDB 存储的 chunk 文本 | 继承 |
| `channel` | "vector" | "bm25" | "memory" | "fused" |
| `score` | float (1 - cosine_distance) | float (BM25) | float (1 - cosine_distance) | float (RRF) |
| `parent_id` | metadata.parent_id | dict.parent_id | None | 继承 |
