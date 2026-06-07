# 知识库 Skill + 向量混合检索改造方案（面向 AI 实施）

## 1. 目标

本方案只解决一件事：在保留 `skill` 作为主检索入口的前提下，为 `knowledge/` 增加可控的向量 + BM25 混合检索补召回能力，并在前端提供重建索引入口。

最终用户链路必须是：

1. 用户提问。
2. 系统先执行 `skill` 检索。
3. 如果 `skill` 证据充分，直接回答。
4. 如果 `skill` 检索不到或只能部分回答，前端明确提示“正在启用向量检索补充证据”。
5. 系统再执行向量 + BM25 混合检索。
6. 融合证据后由回答器统一回答。

## 2. 边界

- 不改 `memory/MEMORY.md` 现有 RAG 逻辑。
- 不保留“主回答 agent 自己边读 skill 边检索边回答”的旧链路。
- `skill` 分支仍然使用大模型通过 `SKILL.md` 和工具执行检索，不改成规则检索器。
- `vector + bm25` 不是第一入口，只在 `skill` 证据不足时启动。
- 当前需求未定义 PDF 的 hybrid 切分规则，因此本方案明确：
  - `pdf` 不进入 hybrid 索引。
  - `pdf` 继续只由 `skill` 分支处理。
  - 如果问题最终被 `skill` 收敛到 PDF 且仍未找到证据，则直接向用户说明当前知识库中未找到完全匹配信息或只能部分回答。

## 3. 现状问题

当前代码中，知识库问题是通过主 agent 在运行时被提示去读取 `skills/rag-skill/SKILL.md` 再自行调用工具完成检索。

这条链路有两个问题：

1. `skill` 检索发生在主回答 agent 推理阶段，不能和前置的检索器形成明确分工。
2. 一旦要加入向量 + BM25，就会出现职责冲突：
   - 不清楚谁负责第一次召回。
   - 不清楚何时切换向量检索。
   - 不清楚前端如何准确展示“skill 没找到，开始补召回”。

因此本次改造的核心不是“多加一个 retriever”，而是把知识库检索正式拆成编排层。

## 4. 最终架构

知识库查询链路改为：

`query -> knowledge_orchestrator -> skill_retriever_agent -> fallback to hybrid_retriever(vector + bm25) -> evidence fusion -> answer_agent`

职责划分如下：

- `skill_retriever_agent`
  - 第一入口。
  - 仍然由大模型驱动。
  - 负责读 `SKILL.md`、读 `data_structure.md`、调用 `read_file / terminal / python_repl` 做渐进式检索。
  - 不负责直接回答用户。
  - 只返回结构化证据和检索状态。

- `hybrid_retriever`
  - 只在 `skill` 检索结果不足时启动。
  - 内部包含两条并行召回分支：
    - 向量检索
    - BM25 检索
  - 只负责从已建索引中召回 `md/json` 证据。

- `knowledge_orchestrator`
  - 负责驱动整个知识库检索流程。
  - 根据 `skill` 检索结果决定是否触发 hybrid。
  - 统一输出最终 evidence。

- `answer_agent`
  - 只消费最终 evidence。
  - 不再直接执行知识库 skill。

## 5. skill 分支的输出契约

`skill_retriever_agent` 不能再输出自然语言答案，必须输出结构化结果。

建议数据结构：

```python
from dataclasses import dataclass
from typing import Literal


@dataclass
class Evidence:
    source_path: str
    source_type: str
    locator: str
    snippet: str
    channel: Literal["skill", "vector", "bm25", "fused"]
    score: float | None
    parent_id: str | None


@dataclass
class SkillRetrievalResult:
    status: Literal["success", "partial", "not_found", "uncertain"]
    evidences: list[Evidence]
    narrowed_paths: list[str]
    narrowed_types: list[str]
    rewritten_queries: list[str]
    searched_paths: list[str]
    reason: str
```

状态含义：

- `success`
  - 已找到足够支持回答的问题证据。
- `partial`
  - 找到部分证据，但不足以支撑完整回答。
- `not_found`
  - 当前 skill 检索范围内没有找到有效证据。
- `uncertain`
  - 找到一些接近证据，但无法确认是否能回答问题。

## 6. hybrid 的启动条件

只有满足以下条件时才启动 `vector + bm25`：

1. `skill_result.status` 属于 `partial / not_found / uncertain`。
2. `skill_result.narrowed_types` 中至少包含 `md` 或 `json`。

不启动 hybrid 的情况：

- `skill_result.status == success`
- `skill` 已将范围收敛到 `excel`
- `skill` 已将范围收敛到 `pdf`

原因：

- `excel` 明确不切分且不适合进入 hybrid 索引。
- `pdf` 当前没有已定义的 hybrid 切分规则。

## 7. 文档切分规则

本次 hybrid 索引只纳入 `md + json`。

### 7.1 Markdown

采用父子文档切分。

- `parent`
  - 一个标题节点及其完整子树。
  - 唯一标识建议为：`path + heading path`
- `child`
  - 该标题节点内按自然段或短段落拆分的检索块。
  - 子块要保留父块引用。

检索策略：

- 向量/BM25 命中 `child`
- 回答阶段回填所属 `parent`
- 对用户展示时优先展示 `parent` 的定位信息和命中片段

### 7.2 JSON

按一个问答对切分。

- 一条问答记录就是一个最小语义单元。
- 不再进一步切分 `question` 和 `answer`。
- `parent` 与 `child` 等同于同一条记录。

建议保留字段：

- `question`
- `answer`
- `label`
- `url`
- `record_id`

### 7.3 Excel

Excel 不切分，也不进入 hybrid 索引。

原因：

- 当前业务语义依赖表结构、列名、筛选条件和聚合过程。
- 切分后会破坏结构语义，召回质量不可控。
- Excel 继续完全交给 `skill` 分支。

### 7.4 PDF

当前不进入 hybrid 索引。

原因：

- 当前需求没有定义 PDF 的 chunk 规则。
- 当前知识库中 PDF 占比高，但没有明确的父子切分标准时，强行入索引会导致召回噪声失控。

## 8. 索引存储与重建

索引统一落到 `backend/storage/knowledge/`。

建议目录结构：

```text
backend/storage/knowledge/
├── manifest.json
├── vector/
├── bm25/
└── derived/
```

含义：

- `manifest.json`
  - 记录每个源文件的索引元信息和指纹。
- `vector/`
  - 向量索引持久化目录。
- `bm25/`
  - BM25 索引持久化目录。
- `derived/`
  - 为 hybrid 准备的中间工件，仅用于 `md/json` 规范化后入索引。

重建原则：

- 明确提供“全量重建知识索引”入口。
- 本次方案不引入增量重建策略。
- 点击重建即执行一次完整、确定性的全量重建。

## 9. 后端接口方案

新增接口：

### 9.1 获取索引状态

`GET /api/knowledge/index/status`

返回：

```json
{
  "ready": true,
  "building": false,
  "last_built_at": 1711111111,
  "indexed_files": 12
}
```

### 9.2 触发重建索引

`POST /api/knowledge/index/rebuild`

返回：

```json
{
  "accepted": true
}
```

要求：

- 如果当前正在重建，接口直接返回“已在进行中”，不要并发启动第二次重建。
- 前端根据状态轮询展示构建进度即可，本方案不强制引入 SSE 进度流。

## 10. 前端方案

### 10.1 重建索引按钮位置

按钮放在现有顶部导航栏控制区，即 `frontend/src/components/layout/Navbar.tsx` 的右侧操作按钮组。

原因：

- 这里已经承载“新会话 / RAG 开关 / 压缩”等系统级操作。
- “重建索引”是知识库级全局操作，不应放在聊天消息区或右侧文件编辑区。

按钮文案：

- 默认：`重建索引`
- 构建中：`索引重建中`

交互要求：

1. 点击按钮后调用 `POST /api/knowledge/index/rebuild`
2. 请求成功后立刻进入“构建中”状态
3. 前端轮询 `GET /api/knowledge/index/status`
4. 构建完成后恢复为默认状态

### 10.2 skill 检索失败提示

当 `skill_retriever_agent` 返回 `partial / not_found / uncertain` 且 orchestrator 决定启动 hybrid 时，前端必须展示明确提示。

提示位置：

- 放在当前 assistant 消息的检索卡片区域，不使用全局 toast。

提示文案：

- `Skill 检索未找到充分证据，正在启用向量检索补充结果。`

这样做的原因：

- 这是当前这条回答链路内部的状态，不是全局系统通知。
- 用户应该能在对应消息下方看到这次回答是如何从 `skill` 切换到向量检索的。

### 10.3 检索卡片展示

当前 `RetrievalCard` 只展示 Memory 召回，后续要改成展示知识检索阶段状态。

建议展示顺序：

1. `skill` 检索结果
2. `fallback` 提示
3. `vector + bm25` 召回结果
4. `fused` 最终证据

## 11. 编排逻辑

```python
skill_result = skill_retriever_agent.retrieve(query)

if skill_result.status == "success":
    final_evidences = skill_result.evidences
    retrieval_notice = None
else:
    if any(file_type in {"md", "json"} for file_type in skill_result.narrowed_types):
        hybrid_result = hybrid_retriever.retrieve(
            query=query,
            path_filters=skill_result.narrowed_paths,
            query_hints=skill_result.rewritten_queries,
        )
        final_evidences = fuse(skill_result.evidences, hybrid_result.evidences)
        retrieval_notice = "Skill 检索未找到充分证据，正在启用向量检索补充结果。"
    else:
        final_evidences = skill_result.evidences
        retrieval_notice = None

answer = answer_agent.answer(
    query=query,
    evidences=final_evidences,
)
```

## 12. 模块拆分

建议新增目录：

```text
backend/knowledge_retrieval/
├── __init__.py
├── types.py
├── indexer.py
├── hybrid_retriever.py
├── skill_retriever_agent.py
├── fusion.py
└── orchestrator.py
```

职责：

- `types.py`
  - 定义 `Evidence / SkillRetrievalResult / HybridRetrievalResult / IndexStatus`
- `indexer.py`
  - 负责 `md/json` 的切分、manifest 写入、向量索引、BM25 索引重建
- `hybrid_retriever.py`
  - 封装向量检索与 BM25 并行召回
- `skill_retriever_agent.py`
  - 创建专用大模型检索 agent，强制读取 `SKILL.md`，只输出结构化结果
- `fusion.py`
  - 做 evidence 去重与 RRF 融合
- `orchestrator.py`
  - 串联 `skill -> fallback hybrid -> fusion -> answer input`

## 13. 需要改动的现有文件

### 后端

- `backend/app.py`
  - 初始化知识索引器和 orchestrator
- `backend/api/chat.py`
  - 支持输出知识检索阶段事件
- `backend/api/files.py`
  - 保存 `knowledge/` 文件后允许后续手动重建；本方案不要求保存即自动重建
- `backend/graph/agent.py`
  - 删除主 agent 的知识库 skill 注入逻辑
  - 将知识库检索改为调用 orchestrator
- `backend/graph/prompt_builder.py`
  - 删除“知识库问题先读 skill 再检索”的运行时 override
  - 改成“优先依据已提供 evidence 回答”

### 前端

- `frontend/src/lib/api.ts`
  - 新增知识索引状态接口和重建接口
- `frontend/src/lib/store.tsx`
  - 新增索引状态管理
  - 新增 knowledge retrieval notice 状态
- `frontend/src/components/layout/Navbar.tsx`
  - 新增“重建索引”按钮
- `frontend/src/components/chat/RetrievalCard.tsx`
  - 支持展示 skill 结果、fallback 提示、vector/bm25 结果、fused 结果
- `frontend/src/components/chat/ChatMessage.tsx`
  - 将新的 retrieval card 信息挂到 assistant 消息上

## 14. 验收标准

以下行为必须全部满足，方案才算完成：

1. 用户发起知识库问题时，系统总是先执行 `skill` 检索。
2. `skill` 检索到充分证据时，不启动向量检索。
3. `skill` 检索不到充分证据时，前端会在当前消息下明确提示正在启用向量检索。
4. hybrid 只索引 `md/json`，不索引 `excel/pdf`。
5. Markdown 按父子文档切分。
6. JSON 按一个问答对切分。
7. Excel 不切分，也不进入 hybrid 索引。
8. 前端提供可点击的“重建索引”按钮，并能看到构建中状态。
9. 主回答 agent 不再自己读取知识库 skill 执行检索。

## 15. 本方案的实施顺序

按最短路径实施：

1. 先完成 `indexer + hybrid_retriever`
2. 再完成 `skill_retriever_agent`
3. 再完成 `orchestrator`
4. 再改主 chat 链路
5. 最后接前端“重建索引”和 fallback 提示

原因：

- 只有先把索引和 hybrid 跑通，`skill` fallback 才有落点。
- 只有 orchestrator 建好，前端提示和事件流才有准确来源。

## 16. 审核点

本方案需要确认的唯一业务前提是：

- 当前版本是否接受：`pdf` 暂不进入 hybrid 索引，仅保留在 `skill` 分支中处理。

如果这个前提成立，本方案即可直接进入实现阶段。
embedding模型用百炼的text-embedding-v4,重排序模型用qwen-rerank
