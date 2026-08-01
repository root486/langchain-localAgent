# 向量索引一致性规则

## 什么操作必须重建索引

| 操作 | 需要重建什么 | 原因 |
|------|-------------|------|
| 修改 `EMBEDDING_MODEL` 或 `EMBEDDING_PROVIDER` | **删除** `storage/knowledge/vector/chroma/` + **清空 PG `memories` 表** **再** rebuild | 新模型的向量维度可能不同，旧索引/旧向量的维度与新模型不兼容 |
| 修改 `_split_markdown` / `_split_json` 逻辑 | rebuild knowledge 索引 | chunk 内容/结构变化，旧索引与新 manifest 不一致 |
| 修改 `_tokenize` 逻辑 | rebuild knowledge 索引 | BM25 tokens 会变，旧 tokens 与新分词器不匹配 |
| 增删 `knowledge/` 下的文件 | rebuild knowledge 索引 | 当前没有自动检测文件变更的机制 |
| 修改 BM25 参数 (k1, b) | rebuild knowledge 索引 | BM25 评分使用硬编码参数 |

## manifest 与 vector 索引不同步的恢复策略

当前 `rebuild_index()` 的执行顺序：
```
_build_documents() → _write_manifest() → _build_bm25_index() → _build_vector_index()
```

**风险**：如果在 `_write_manifest()` 之后、`_build_vector_index()` 之前崩溃：
- manifest 是新的，vector 索引是旧的
- 下次启动：`_load_manifest()` 加载新 manifest，`_load_vector_index()` 加载旧 vector
- 结果：BM25 拿新数据查，向量拿旧数据查 → **检索结果不一致**

**恢复方法**：
1. 调用 `POST /api/knowledge/index/rebuild` 触发完整重建
2. 或者手动删除 `storage/knowledge/` 下所有内容后重启

**建议改进**：增加索引版本号/模型签名校验，加载时检测不一致自动触发 rebuild。

## Embedding 模型变更检查清单

- [ ] 确认新模型的向量维度与旧模型相同（否则必须全量重建）
- [ ] 删除 `backend/storage/knowledge/vector/chroma/` 目录
- [ ] 清空 PG `memories` 表（长期记忆向量，维度会变）
- [ ] 更新 `.env` 中的 `EMBEDDING_MODEL` / `EMBEDDING_PROVIDER` / `EMBEDDING_API_KEY`
- [ ] 重启后端进程
- [ ] 验证：`GET /api/knowledge/index/status` 返回 `vector_ready: true`
- [ ] 验证：发送知识库查询，确认检索结果非空
