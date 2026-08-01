"""RAGAS 0.4.3 ↔ langchain-community 1.x 兼容补丁。

ragas.llms.base 顶层执行 `from langchain_community.chat_models.vertexai import ChatVertexAI`，
但 langchain-community 1.x 已移除该模块（社区版已 sunset）。
本模块用 `langchain_google_vertexai` 桥接，让 `import ragas` 不崩。

**必须在任何 `import ragas` 之前导入本模块。** evaluate_ragas.py 顶部已处理。
"""
import sys

import langchain_community.chat_models
import langchain_google_vertexai

langchain_community.chat_models.vertexai = langchain_google_vertexai  # type: ignore[attr-defined]
sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai
