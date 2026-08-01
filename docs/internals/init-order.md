# 初始化顺序与单例依赖

## 模块级单例

项目中所有核心模块都使用模块级单例模式（文件底部实例化）：

```python
# knowledge_retrieval/indexer.py
knowledge_indexer = KnowledgeIndexer()

# knowledge_retrieval/orchestrator.py
knowledge_orchestrator = KnowledgeOrchestrator()

# knowledge_retrieval/hybrid_retriever.py
hybrid_retriever = HybridRetriever()

# graph/memory_store.py
memory_store = MemoryStore()

# graph/agent.py
agent_manager = AgentManager()
```

## 初始化顺序

`app.py lifespan` 中的初始化序列（**顺序不能随意调换**）：

```
1. get_settings()              → 加载 .env 配置（@lru_cache，只执行一次）
2. refresh_snapshot()           → 生成 SKILLS_SNAPSHOT.md
3. _init_mcp_tools()           → 按 key 加载 MCP 工具（Tavily stdio + 高德 streamable http，逐个 server 用 _load_server_tools 带超时+重试；单个失败不影响其它）
4. agent_manager.initialize(base_dir, mcp_tools) → 内部依次执行:
   4a. SessionManager(base_dir) → 创建 sessions 目录
   4b. get_all_tools(base_dir)  → 实例化工具
   4c. knowledge_orchestrator.configure(base_dir, model_builder)
       → 配置检索编排器（Multi-Query 展开依赖 model_builder）
5. memory_store.configure()    → PG 连接池 + 建 memories 表（text + float8[] embedding；PG 不可用时自动降级，is_ready()=False）
6. knowledge_indexer.configure() → 创建 storage/knowledge 子目录 + _load_manifest + _load_vector_index
7. knowledge_indexer.rebuild_index() → 重建 knowledge 索引
```

**依赖关系**：
- `knowledge_orchestrator.configure` 在 `agent_manager.initialize` 内部调用，不能提前
- `knowledge_indexer.rebuild_index` 依赖 `get_settings()` 返回正确的 embedding 配置
- 所有单例都是 import 时创建空实例，lifespan 时才 configure，**不要在 configure 之前调用业务方法**

## 新增单例的规范

1. 在模块底部创建空实例
2. 在 `app.py lifespan` 中适当位置调用 `.configure()`
3. 确保 configure 顺序满足依赖关系
4. 在业务方法开头加 `if self.base_dir is None: raise RuntimeError("not configured")` 防御
