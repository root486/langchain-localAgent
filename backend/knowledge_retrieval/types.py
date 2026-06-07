from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


RetrievalChannel = Literal["memory", "skill", "vector", "bm25", "fused"]
RetrievalKind = Literal["memory", "knowledge"]


@dataclass
class Evidence:
    source_path: str
    source_type: str
    locator: str
    snippet: str
    channel: RetrievalChannel
    score: float | None = None
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalStep:
    kind: RetrievalKind
    stage: str
    title: str
    message: str = ""
    results: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "stage": self.stage,
            "title": self.title,
            "message": self.message,
            "results": [item.to_dict() for item in self.results],
        }


@dataclass
class SkillRetrievalResult:
    status: Literal["success", "partial", "not_found", "uncertain"]
    evidences: list[Evidence] = field(default_factory=list)
    narrowed_paths: list[str] = field(default_factory=list)
    narrowed_types: list[str] = field(default_factory=list)
    rewritten_queries: list[str] = field(default_factory=list)
    searched_paths: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "evidences": [item.to_dict() for item in self.evidences],
            "narrowed_paths": list(self.narrowed_paths),
            "narrowed_types": list(self.narrowed_types),
            "rewritten_queries": list(self.rewritten_queries),
            "searched_paths": list(self.searched_paths),
            "reason": self.reason,
        }


@dataclass
class HybridRetrievalResult:
    vector_evidences: list[Evidence] = field(default_factory=list)
    bm25_evidences: list[Evidence] = field(default_factory=list)


@dataclass
class OrchestratedRetrievalResult:
    status: Literal["success", "partial", "not_found", "uncertain"]
    evidences: list[Evidence] = field(default_factory=list)
    steps: list[RetrievalStep] = field(default_factory=list)
    fallback_used: bool = False
    reason: str = ""


@dataclass
class IndexStatus:
    ready: bool
    building: bool
    last_built_at: float | None
    indexed_files: int
    vector_ready: bool
    bm25_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
