# 优化后验证步骤

## 每次修改后必做的验证

- [ ] **启动验证**：后端能正常启动，无报错
- [ ] **健康检查**：`GET /health` 返回 `{"status": "ok"}`
- [ ] **索引状态**：`GET /api/knowledge/index/status` 返回 `ready: true`
- [ ] **普通对话**：发送非知识库问题，收到正常 token 流 + done 事件
- [ ] **知识库查询**：发送包含"知识库"的问题，收到 retrieval + token + done 事件
- [ ] **前端渲染**：RetrievalCard 正确显示检索步骤，ThoughtChain 正确显示工具调用

## 修改检索逻辑后的验证

- [ ] **检索正确性**：用已知 query 验证检索结果是否符合预期
- [ ] **索引完整性**：manifest.json 中的 chunk 数量 == ChromaDB collection 的 count（粗略检查）
- [ ] **PDF 索引**：知识库含 `.pdf` 时 rebuild 后 manifest 出现 `source_type="pdf"` 的 chunk；扫描版（纯图片）PDF 自动跳过不崩溃
- [ ] **BM25 可用**：在 `vector_ready=false` 的情况下，BM25 检索仍然能返回结果
- [ ] **RRF 融合**：两路（vector/bm25）证据融合后无重复、无丢失

## 修改 Embedding/索引逻辑后的验证

- [ ] 执行完整的 rebuild：`POST /api/knowledge/index/rebuild`
- [ ] 等待 building 状态变为 false
- [ ] 验证 vector_ready 和 bm25_ready 都为 true
- [ ] 发送知识库查询，确认检索结果非空且相关

## 修改前后端协议后的验证

- [ ] 后端新增/修改事件 → 前端对应 onEvent 分支已更新
- [ ] 后端修改 dataclass 字段 → 前端对应 type 定义已更新
- [ ] 后端新增枚举值 → 前端 STEP_META / 类型定义已同步
- [ ] SSE 事件流完整：token → ... → done，无事件丢失

## 回归测试

```bash
# BM25 离线评估（不需要 LLM，3 秒跑完）
python backend/scripts/evaluate_faq_retrieval.py
```

修改检索逻辑后，至少运行 BM25 离线评估确认不退化。
