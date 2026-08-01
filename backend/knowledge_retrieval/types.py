from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


RetrievalChannel = Literal["memory", "vector", "bm25", "fused"]#检索渠道
RetrievalKind = Literal["memory", "knowledge"]#检索类型

#检索证据（前端使用）
@dataclass
class Evidence:
    source_path: str  # 来源文件路径
    source_type: str  # 来源文件类型（md/json等）
    locator: str  # 在文件中的定位符
    snippet: str  # 文本片段内容
    channel: RetrievalChannel  # 检索渠道（memory/vector/bm25/fused）
    score: float | None = None  # 相关度分数
    parent_id: str | None = None  # 父级证据ID

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

#检索步骤（前端使用）
@dataclass
class RetrievalStep:
    kind: RetrievalKind  # 检索类型（memory/knowledge）
    stage: str  # 检索阶段（vector/bm25/rerank/fused/memory）
    title: str  # 步骤标题
    message: str = ""  # 步骤描述
    results: list[Evidence] = field(default_factory=list)  # 该步骤的检索证据列表

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "stage": self.stage,
            "title": self.title,
            "message": self.message,
            "results": [item.to_dict() for item in self.results],
        }

#混合检索结果（向量证据+BM25证据）
@dataclass
class HybridRetrievalResult:
    vector_evidences: list[Evidence] = field(default_factory=list)  # 向量检索的证据
    bm25_evidences: list[Evidence] = field(default_factory=list)  # BM25检索的证据

#编排后的最终结果（所有步骤+所有证据）
@dataclass
class OrchestratedRetrievalResult:
    status: Literal["success", "not_found"]  # 检索状态（旧窄路径 skill_retriever_agent 的 partial/uncertain 已移除）
    evidences: list[Evidence] = field(default_factory=list)  # 所有证据的合集
    steps: list[RetrievalStep] = field(default_factory=list)  # 所有检索步骤
    reason: str = ""  # 检索结果说明

#知识库索引的当前状态（前端使用）
@dataclass
class IndexStatus:
    ready: bool  # 索引是否就绪
    building: bool  # 是否正在建索引
    last_built_at: float | None  # 上次建索引的时间戳
    indexed_files: int  # 已索引的文件数量
    vector_ready: bool  # 向量索引是否就绪
    bm25_ready: bool  # BM25索引是否就绪

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)