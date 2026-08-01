# 并发安全与状态机

## rebuild 期间的查询行为

```
KnowledgeIndexer:
  rebuild_index() 持有 self._lock，设置 self._building = True
  │
  ├─ retrieve_vector() / retrieve_bm25()
  │   不加锁，开头调用 _ensure_loaded()
  │   如果 _documents 为空或 _vector_index 为 None → 尝试从磁盘加载
  │   ⚠️ 如果磁盘上的 manifest.json 正在被写入，可能读到半写状态
  │
  └─ is_building() → 前端用于显示"重建中"状态
```

**规则**：
- 不要在 `rebuild_index()` 执行期间修改 retrieve 逻辑（可能导致读到中间状态）
- 不要删除 `_building` 标志的检查逻辑（前端依赖它轮询状态）

## 前端 building 状态轮询的边界条件

```typescript
// store.tsx: 只在 building=true 时轮询，3 秒间隔
useEffect(() => {
  if (!knowledgeIndexStatus?.building) return;
  const timer = setInterval(() => {
    getKnowledgeIndexStatus().then(status => setKnowledgeIndexStatus(status));
  }, 3000);
  return () => clearInterval(timer);
}, [knowledgeIndexStatus?.building]);
```

**风险**：如果后端 rebuild 异常退出（`_building` 没有被重置为 False），前端会永远停在"重建中"状态。

**规则**：
- `rebuild_index()` 的 `finally` 块中 `_building = False` 绝对不能删除
- 如果优化 rebuild 逻辑，确保所有异常路径都重置 `_building`

## rebuild 触发方式

| 触发方式 | 线程 | 是否阻塞 |
|----------|------|---------|
| 启动时 `app.py lifespan` | 主线程 | 阻塞启动，直到完成 |
| `POST /api/knowledge/index/rebuild` | `asyncio.to_thread` 新线程 | 不阻塞 API 请求 |
