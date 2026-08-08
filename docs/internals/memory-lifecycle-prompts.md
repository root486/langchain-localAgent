# 长期记忆生命周期：抽取器 / 整合决策器 / 遗忘策略 提示词（DEPRECATED）

> ⚠️ **已废弃（2026-08-01）**：长期记忆已重构为简单 ChromaDB 单 collection 方案（见 `graph/memory_store.py`），
> 整合决策器 / 遗忘规则 / scope-category-status 枚举全部删除。
> 现保留的只有「抽取器」概念：一次 LLM 调用从对话提取纯 text 事实数组
> （`EXTRACTOR_SYSTEM_PROMPT`，输出 `{"memories": ["..."]}`），写入入口 `MemoryStore.remember()`，
> 触发在 `api/chat.py`（`_schedule_memory_extraction`，游标增量抽取）。
> 去重由嵌入余弦 `DEDUP_THRESHOLD=0.93` 完成，不再走 LLM 决策器；遗忘由 `_prune` 行数上限（2000）承担。
> 本文档保留仅供历史查阅。

---

## 一、抽取器（Write 阶段）— 对话 → 原子事实

```text
# 角色
你是长期记忆抽取器，负责从对话中提炼值得跨会话记住的稳定事实。

# 输入
一份对话记录（按顺序列出用户与助手消息），以及当前已有的记忆摘要（可选）。

# 抽取原则
1. 只抽「稳定、可复用」的信息：
   - 用户偏好（语言、格式、工作方式、工具偏好）
   - 个人/团队属性（职业、行业、技能、角色）
   - 项目/领域事实（项目背景、技术选型、业务规则）
   - 决策与目标（明确了的选择、正在追求的目标）
   - 长期约束（红线、必须遵守的规则）
2. 不抽：寒暄、一次性请求、纯提问、情绪、偶然的临时信息。
3. 一条记忆 = 一个原子陈述（单一事实），用陈述句、第三人称，不含多个要点。
4. 与已有记忆重复的内容不重复抽取；用户最新表述与旧信息冲突时，以最新为准。
5. 对话中没有值得记住的内容 → 返回空数组。

# 输出（严格 JSON，不要任何多余文字）
{
  "memories": [
    {"scope": "user", "category": "preference", "text": "User 偏好英文交流", "confidence": 0.9}
  ]
}
- scope ∈ user | project | domain | agent
- category ∈ preference | fact | decision | goal | constraint
- confidence: 0~1，抽取器对「该信息确实稳定可复用」的确信度

# 对话记录
{messages_json}

# 已有记忆摘要（可选，帮助避免重复抽取）
{memory_summary}
```

### 调用约定
- `messages_json`：近期对话消息数组 `[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]`（建议最近 10~20 条，若过短可并入会话压缩摘要）。
- `memory_summary`：已有记忆的压缩摘要（可选；由检索现有 memories 后让模型概括，或直接省略）。
- 解析：`json.loads` 提取 `memories`，逐条过滤 `text` 非空、`scope`/`category` 在枚举内；解析失败/空数组 → 本轮不写入。
- 建议用轻量模型（`config.summary_model`，如 deepseek-v4-flash），`temperature=0`。

---

## 二、整合决策器（Consolidate 阶段）— 新事实 × 已有记忆 → ADD/UPDATE/DELETE/NOOP

```text
# 角色
你是记忆整合决策器。系统会给出若干条「新抽取的记忆」，每条附带与之语义相近的「已有记忆」。
请对每条新记忆决定一个操作：ADD / UPDATE / DELETE / NOOP。

# 操作定义
- ADD    : 新记忆与任何已有记忆都无实质重叠 → 新增一条。
- UPDATE : 某条已有记忆主题相同但内容不同/过时，新记忆应取代它 → 修改该记忆（合并成信息更完整的版本）。
- DELETE : 新记忆与某条已有记忆直接矛盾并推翻它 → 将该记忆标记为 superseded（不再被检索）。
- NOOP   : 已有记忆已完整覆盖该信息 → 不做任何操作。

# 规则
1. 判断依据是「语义等价」，不是字面相同。
2. 冲突时以新记忆为准（用户最新表达优先于旧记忆）。
3. UPDATE / DELETE 必须给出 target_id（指向已有记忆的 id）。
4. UPDATE 必须给出 new_text（合并后信息更完整的版本）。
5. 每个 candidate_index 恰好输出一条 operation。
6. 输出必须严格 JSON，不要任何多余文字。

# 输出格式
{
  "operations": [
    {"candidate_index": 0, "operation": "ADD", "target_id": null, "new_text": null, "reason": "全新信息，无冲突"}
  ]
}

# 待处理的新记忆（每条附带相近已有记忆）
1. 新记忆: "User 偏好英文交流" (scope=user, category=preference)
   相近已有:
     - id=12: "User 偏好中文交流" (scope=user, category=preference)
     - id=8 : "用户从事跨境电商" (scope=user, category=fact)
2. 新记忆: "项目后端改为 PostgreSQL 存记忆" (scope=project, category=decision)
   相近已有: (无)
...
```

### 调用约定
- 输入组装：对每条 candidate 事实嵌入后，在 ChromaDB `memory_facts` 检索 top-3 相近已有记忆（`status='active'`），连同 `id`、`text` 一并注入。
- 解析 `operations`，按 `candidate_index` 对齐；缺省/非法 → 该条按 ADD 兜底。
- 建议 `temperature=0`，轻量模型。

---

## 三、遗忘策略（Forget 阶段）— 规则，非 LLM

1. 检索只返回 `status='active'` 且 `score >= MIN_SCORE`（0.4）。
2. 每次检索命中 → 更新该记忆的 `last_used_at`。
3. 定期任务（每次启动时执行一次即可，或每日）：
   - `last_used_at` 为空或距今 > 180 天 → `status='archived'`；
   - 已被 DELETE/UPDATE 覆盖的旧版本 → `status='superseded'`。
4. 行数上限（建议 5000）：超限时优先淘汰 `archived` 中 `created_at` 最早的记录。
5. （可选增强）检索排序时对 `last_used_at` 做轻微时间衰减加权，避免老记忆长期霸榜。

---

## 四、集成点（代码侧）

### `graph/memory_store.py` 新增方法
```
write_from_session(messages: list[dict]) -> int   # 返回实际写入条数
```
1. 组装抽取器输入 → 调 LLM（复用 `agent_manager._build_chat_model()` 或 `config.summary_model`）→ 得候选事实；
2. 逐条候选：`EmbeddingClient().embed_one(text)` → ChromaDB 检索 top-3 相近已有记忆；
3. 组装决策器输入 → 一次 LLM 调用得 operations；
4. 执行：
   - `ADD`    → `self.add(text, scope, category)`（已实现）；
   - `UPDATE` → 更新 PG 行 `text` + 重新嵌入 + 覆盖 ChromaDB 文档；
   - `DELETE` → PG 行 `status='superseded'`（ChromaDB 端可保留或删除，检索端按 PG status 过滤）；
   - `NOOP`   → 跳过。

### 触发
- `api/chat.py`：仿现有 `_schedule_compression()` 加 `_schedule_memory_extraction(session_id)` 后台任务，
  在会话压缩时（`_run_compression` 内）或会话空闲时，用最近消息调 `write_from_session`。
- 注意：与压缩任务共享后台线程池，失败仅打日志，不影响 SSE 响应。

### 校验
- 写入后 `memory_store.status()["count"]` 增加；
- 再问相关问题能召回新事实；改口的事实不再自相矛盾（UPDATE/DELETE 生效）。
