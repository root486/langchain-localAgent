# SSE 事件协议

## 完整事件列表

后端 `agent.py` / `orchestrator.py` yield 的事件 → 前端 `store.tsx` onEvent 处理：

| 事件名 | 后端 yield 来源 | 前端处理 | 必须包含的字段 |
|--------|----------------|----------|---------------|
| `token` | AgentManager | 拼接到 assistant.content | `content: str` |
| `tool_start` | AgentManager | 追加一个 output="" 的 toolCall | `tool: str`, `input: str` |
| `tool_end` | AgentManager | 填充最后一个 toolCall 的 output | `output: str` |
| `retrieval` | AgentManager (memory) / KnowledgeOrchestrator (knowledge) | normalizeRetrievalStep → 追加到 retrievalSteps | `kind`, `stage`, `title`, `message`, `results[]` |
| `new_response` | AgentManager (普通对话) | 创建新的 assistant 消息对象 | 无 |
| `done` | AgentManager | 兜底填充 content（如果为空） | `content: str` |
| `title` | AgentManager (首条消息后) | 刷新会话列表 | `session_id: str`, `title: str` |
| `error` | chat.py 异常处理 | 显示错误 | `error: str` |
| `orchestrated_result` | KnowledgeOrchestrator | **前端不处理**（被 agent.py 消费） | — |

## 事件顺序约束

1. `tool_start` 必须先于对应的 `tool_end`（前端靠"最后一个 toolCall"配对）
2. `done` 必须是每个请求流的最后一个事件
3. `title` 事件只在 `done` 之后、且是首条用户消息时发送
4. `retrieval` 事件在 `token` 事件之前发送（先展示检索轨迹，再展示回答）

## 修改规则

- **新增事件类型**：后端加新事件 → 前端 `onEvent` 必须加对应分支，否则事件被静默忽略
- **删除事件类型**：后端删除事件 → 前端对应分支变成死代码，但不崩溃
- **改事件名**：前后端必须同步修改，否则前端走不到对应分支，消息丢失
