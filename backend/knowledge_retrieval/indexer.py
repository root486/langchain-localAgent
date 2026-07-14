from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from llama_index.core import Document, Settings as LlamaSettings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding

from config import get_settings
from knowledge_retrieval.types import Evidence, IndexStatus


# Monkey-patch: 允许非 OpenAI 官方的 embedding 模型名（如 Bailian 的 text-embedding-v4）
# LlamaIndex 的 OpenAIEmbedding 会在构造时校验模型名是否在 OpenAIEmbeddingModelType 枚举中，
# 导致第三方兼容 API 的模型名被拒绝。此补丁让 get_engine 在遇到未知模型名时直接返回模型名。
def _patch_llama_index_get_engine() -> None:
    import llama_index.embeddings.openai.base as _embed_base
    _orig = _embed_base.get_engine

    def _patched(mode: str, model: str, mode_model_dict: dict) -> str:
        try:
            return _orig(mode, model, mode_model_dict)
        except ValueError:
            return model

    _embed_base.get_engine = _patched


_patch_llama_index_get_engine()


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
ALNUM_PATTERN = re.compile(r"[A-Za-z0-9_]+")
CHINESE_BLOCK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
# \u53e5\u5b50\u8fb9\u754c\uff1a\u4e2d\u6587\u6807\u70b9\u6216\u6362\u884c\u540e\u7684\u975e\u7a7a\u884c\u9996
SENT_BOUNDARY = re.compile(r"[\u3002\uff01\uff1f\uff1b\n](?=\S)")
# \u4e0a\u4e0b\u6587\u7a97\u53e3\u4e0a\u9650\uff08\u5b57\u7b26\u6570\uff09\uff1aP95 \u7684 parent_text \u4e3a 1357\uff0c\u53d6 1500 \u8986\u76d6\u7edd\u5927\u90e8\u5206\u573a\u666f
MAX_CONTEXT_CHARS = 1500
# \u6700\u5c0f chunk \u5927\u5c0f\uff1a\u4f4e\u4e8e\u6b64\u503c\u7684\u76f8\u90bb\u540c parent chunk \u4f1a\u88ab\u5408\u5e76
MIN_CHUNK_SIZE = 60


class KnowledgeIndexer:
    def __init__(self) -> None:
        self.base_dir: Path | None = None # 项目根目录
        self._vector_index: VectorStoreIndex | None = None# LlamaIndex向量索引对象
        self._documents: list[dict[str, Any]] = []# 所有chunk的列表
        self._lock = threading.Lock()  # 线程锁（防止并发重建）
        self._building = False # 是否正在建索引
        self._last_built_at: float | None = None # 上次建索引时间
        self._avg_doc_length = 0.0 # BM25用：平均文档长度
        self._document_frequencies: Counter[str] = Counter()# BM25用：每个词出现在多少文档中
        self._vector_ready = False # 向量索引是否就绪
        self._bm25_ready = False # BM25索引是否就绪

    #configure 被调用后，如果磁盘上有之前建的索引，直接加载，不用重建。
    def configure(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True) # storage/knowledge/
        self._vector_dir.mkdir(parents=True, exist_ok=True) # storage/knowledge/vector/
        self._bm25_dir.mkdir(parents=True, exist_ok=True) # storage/knowledge/bm25/
        self._derived_dir.mkdir(parents=True, exist_ok=True) # storage/knowledge/derived/
        self._load_manifest() # 从磁盘加载chunk数据
        self._load_vector_index() # 从磁盘加载向量索引

    @property
    def _knowledge_dir(self) -> Path:
        if self.base_dir is None:
            raise RuntimeError("KnowledgeIndexer is not configured")
        return self.base_dir / "knowledge"

    @property
    def _storage_dir(self) -> Path:
        if self.base_dir is None:
            raise RuntimeError("KnowledgeIndexer is not configured")
        return self.base_dir / "storage" / "knowledge"

    @property
    def _manifest_path(self) -> Path:
        return self._storage_dir / "manifest.json"

    @property
    def _vector_dir(self) -> Path:
        return self._storage_dir / "vector"

    @property
    def _bm25_dir(self) -> Path:
        return self._storage_dir / "bm25"

    @property
    def _derived_dir(self) -> Path:
        return self._storage_dir / "derived"

    def _supports_embeddings(self) -> bool:
        return bool(get_settings().embedding_api_key)

    def _build_embed_model(self) -> OpenAIEmbedding:
        settings = get_settings()
        return OpenAIEmbedding(
            api_key=settings.embedding_api_key,
            api_base=settings.embedding_base_url,
            model=settings.embedding_model,
            embed_batch_size=10,  # 百炼 text-embedding-v4 限制最大 10
        )

    def status(self) -> IndexStatus:
        return IndexStatus(
            ready=bool(self._documents) and (self._vector_ready or self._bm25_ready),
            building=self._building,
            last_built_at=self._last_built_at,
            indexed_files=len({item["source_path"] for item in self._documents}),
            vector_ready=self._vector_ready,
            bm25_ready=self._bm25_ready,
        )

    def is_building(self) -> bool:
        return self._building
    #建索引的主要方法，先读文件切分成chunk，然后持久化到manifest.json，再统计BM25数据，最后建向量索引
    def rebuild_index(self) -> None:
        if self.base_dir is None:
            return

        with self._lock: # 加锁，防止并发重建
            self._building = True
            try:
                self._documents = self._build_documents()# 读文件+切分
                self._write_manifest()  # 持久化到manifest.json
                self._prepare_bm25_stats()# 统计BM25数据
                self._build_vector_index()   # 建向量索引
                self._last_built_at = time.time()
            finally:
                self._building = False

    def _relative_path(self, path: Path) -> str:
        if self.base_dir is None:
            return str(path)
        return str(path.relative_to(self.base_dir)).replace("\\", "/")
    # 从knowledge目录下的所有文件中读取内容，切分成chunk    
    def _build_documents(self) -> list[dict[str, Any]]:
        if not self._knowledge_dir.exists():
            return []

        documents: list[dict[str, Any]] = []
        for path in sorted(self._knowledge_dir.rglob("*")):# 递归遍历knowledge/下所有文件
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".md":
                documents.extend(self._split_markdown(path))# md文件按标题切
            elif suffix == ".json":
                documents.extend(self._split_json(path))# json文件按记录切
        return documents

    def _split_markdown(self, path: Path) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8")
        source_path = self._relative_path(path)
        sections: list[tuple[list[str], list[str]]] = []
        heading_stack: list[str] = []
        current_lines: list[str] = []

        def flush_section() -> None:
            if not current_lines:
                return
            heading_path = heading_stack[:] if heading_stack else [path.stem]
            sections.append((heading_path, current_lines[:]))

        for raw_line in text.splitlines():
            match = HEADING_PATTERN.match(raw_line)
            if not match:
                current_lines.append(raw_line)
                continue
            flush_section()
            current_lines = [raw_line]
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)

        flush_section()
        if not sections:
            sections = [([path.stem], text.splitlines())]

        raw_chunks: list[dict[str, Any]] = []
        for section_index, (heading_path, lines) in enumerate(sections, start=1):
            section_text = "\n".join(lines).strip()
            if not section_text:
                continue
            parent_id = f"{source_path}::{' > '.join(heading_path)}"
            # ---------- 优化 3：过长片段拆分 ----------
            # 不在 1200 字符处硬切，而是找到最近的句子边界（。！？；换行）再切
            slices = self._split_long_text(section_text)
            for si, slice_text in enumerate(slices, start=1):
                locator = f"{' > '.join(heading_path)}"
                if len(slices) > 1:
                    locator = f"{locator} (片段 {si}/{len(slices)})"
                raw_chunks.append(
                    {
                        "parent_id": parent_id,
                        "source_path": source_path,
                        "source_type": "md",
                        "locator": locator,
                        "text": slice_text,
                        "parent_text": section_text,
                        "section_index": section_index,
                    }
                )
        # ---------- 优化 4：过短片段合并 ----------
        # 同 parent 内，相邻 chunk 如果任一小于 MIN_CHUNK_SIZE，合并它们
        return self._merge_short_chunks(raw_chunks, source_path)


    def _split_json(self, path: Path) -> list[dict[str, Any]]:
        source_path = self._relative_path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []

        chunks: list[dict[str, Any]] = []
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            label = str(item.get("label", "")).strip()
            url = str(item.get("url", "")).strip()
            if not question and not answer:
                continue

            record_id = str(item.get("record_id") or item.get("id") or index)
            parent_id = f"{source_path}::record::{record_id}"
            parts = [f"Question: {question}", f"Answer: {answer}"]
            if label:
                parts.append(f"Label: {label}")
            if url:
                parts.append(f"URL: {url}")
            parent_text = "\n".join(parts)
            # 长 answer 按句子边界拆分子 chunk
            slices = self._split_long_text(parent_text)
            for si, slice_text in enumerate(slices, start=1):
                locator = f"记录 {record_id}"
                if len(slices) > 1:
                    locator = f"{locator} (片段 {si}/{len(slices)})"
                chunks.append(
                    {
                        "doc_id": f"{parent_id}::child::{si}",
                        "parent_id": parent_id,
                        "source_path": source_path,
                        "source_type": "json",
                        "locator": locator,
                        "text": slice_text,
                        "parent_text": parent_text,
                        "record_id": record_id,
                    }
                )
        return chunks
    # ---------- 父子 chunk 工具方法 ----------

    @staticmethod
    def _split_long_text(text: str, chunk_size: int = 1200) -> list[str]:
        """在句子边界处切分长文本，避免在句中硬切。

        找到 chunk_size 范围内最后一个句子终止符（。！？；\\n），在此处切开。
        如果找不到边界，回退到 chunk_size 处硬切（兼容超长无标点的文本）。
        """
        if len(text) <= chunk_size:
            return [text]

        slices: list[str] = []
        pos = 0
        while pos < len(text):
            end = min(pos + chunk_size, len(text))
            if end >= len(text):
                slices.append(text[pos:].strip())
                break
            # 在 [pos, end] 范围内找最后一个句子边界
            window = text[pos:end]
            boundary = -1
            for m in SENT_BOUNDARY.finditer(window):
                boundary = m.start() + 1  # 在标点符号之后切
            if boundary > chunk_size // 3:
                # 找到了合理的边界（至少超过 1/3 chunk_size，避免太碎）
                end = pos + boundary
            slices.append(text[pos:end].strip())
            pos = end
        return [s for s in slices if s]

    def _merge_short_chunks(
        self, chunks: list[dict[str, Any]], source_path: str
    ) -> list[dict[str, Any]]:
        """合并同 parent 内过短的相邻 chunk。

        如果 chunk A 或 chunk B 任一小于 MIN_CHUNK_SIZE，将它们合并。
        合并后的 chunk 保留靠前的 locator，用于追溯位置。
        """
        if len(chunks) <= 1:
            return self._assign_doc_ids(chunks, source_path)

        merged: list[dict[str, Any]] = []
        prev = chunks[0]
        for cur in chunks[1:]:
            same_parent = prev["parent_id"] == cur["parent_id"]
            too_short = (
                len(prev["text"]) < MIN_CHUNK_SIZE
                or len(cur["text"]) < MIN_CHUNK_SIZE
            )
            if same_parent and too_short:
                # 合并：文本拼接，保留 prev 的 locator
                prev["text"] = prev["text"] + "\n\n" + cur["text"]
            else:
                merged.append(prev)
                prev = cur
        merged.append(prev)
        return self._assign_doc_ids(merged, source_path)

    @staticmethod
    def _assign_doc_ids(
        chunks: list[dict[str, Any]], source_path: str
    ) -> list[dict[str, Any]]:
        """为合并后的 chunk 重新分配 doc_id，保持唯一性。"""
        for i, c in enumerate(chunks, start=1):
            c["doc_id"] = f"{c['parent_id']}::child::{i}"
        return chunks
    # ---------- 持久化 ----------
    def _write_manifest(self) -> None:
        payload = {
            "built_at": time.time(),
            "documents": self._documents,
        }
        self._manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_manifest(self) -> None:
        if not self._manifest_path.exists():
            self._documents = []
            self._bm25_ready = False
            return
        try:
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._documents = []
            self._bm25_ready = False
            return
        self._documents = list(payload.get("documents", []))
        self._last_built_at = payload.get("built_at")
        self._prepare_bm25_stats()
    #为 BM25 检索准备三个统计数据：平均文档长度、文档频率、词频。
    def _prepare_bm25_stats(self) -> None:
        if not self._documents:
            self._avg_doc_length = 0.0#平均文档长度（所有chunk的平均token数）
            self._document_frequencies = Counter()#每个词在多少个文档中出现过（词频）
            self._bm25_ready = False
            return

        self._document_frequencies = Counter()
        doc_lengths: list[int] = []
        for item in self._documents:
            tokens = self._tokenize(str(item.get("text", "")))#分词，生成token列表
            item["tokens"] = tokens
            doc_lengths.append(len(tokens))
            for token in set(tokens):
                self._document_frequencies[token] += 1

        self._avg_doc_length = sum(doc_lengths) / max(1, len(doc_lengths))
        self._bm25_ready = True

    def _build_vector_index(self) -> None:
        if not self._supports_embeddings() or not self._documents:
            self._vector_index = None
            self._vector_ready = False
            return

        try:
            LlamaSettings.embed_model = self._build_embed_model()#配置Embedding模型
            print(f"[索引] 正在向量化 {len(self._documents)} 个 chunk（批大小=50）...")
            #把每个 chunk 转成 LlamaIndex 的 Document 对象。text 是要转向量的文本，metadata 是附带信息，检索结果里会原样返回。
            documents = [
                Document(
                    text=str(item["text"]),# chunk的文本内容
                    metadata={
                        "doc_id": item["doc_id"],
                        "parent_id": item["parent_id"],
                        "source_path": item["source_path"],
                        "source_type": item["source_type"],
                        "locator": item["locator"],
                    },
                )
                for item in self._documents
            ]
            self._vector_index = VectorStoreIndex.from_documents(documents)#LlamaIndex 对每个 Document 的 text 调用 Embedding 模型，转成向量，建索引。
            self._vector_index.storage_context.persist(persist_dir=str(self._vector_dir))#把索引存到 storage/knowledge/vector/ 目录，下次启动不用重建。
            self._vector_ready = True
            print(f"[索引] 向量化完成，已持久化到 {self._vector_dir}")
        except Exception as exc:
            print(f"[索引] 向量化失败: {exc}")
            import traceback
            traceback.print_exc()
            self._vector_index = None
            self._vector_ready = False

    def _load_vector_index(self) -> None:
        if not self._supports_embeddings():
            self._vector_index = None
            self._vector_ready = False
            return
        #vector/ 目录是空的 → 没有历史索引，跳过。
        if not list(self._vector_dir.glob("*")):
            self._vector_index = None
            self._vector_ready = False
            return
        try:
            LlamaSettings.embed_model = self._build_embed_model()#配置 Embedding 模型（检索时需要用它把 query 也转向量）
            storage_context = StorageContext.from_defaults(persist_dir=str(self._vector_dir))
            self._vector_index = load_index_from_storage(storage_context)#从 storage/knowledge/vector/ 目录加载索引
            self._vector_ready = True
        except Exception:
            self._vector_index = None
            self._vector_ready = False

    def _ensure_loaded(self) -> None:
        if not self._documents:
            self._load_manifest()
        if self._vector_index is None and self._supports_embeddings():
            self._load_vector_index()
    #判断“这个 chunk 属不属于skill agent指定的目录”
    def _matches_path_filters(self, source_path: str, path_filters: list[str] | None) -> bool:
        if not path_filters:
            return True
        normalized = source_path.replace("\\", "/")
        for path_filter in path_filters:
            candidate = path_filter.replace("\\", "/").strip()
            if not candidate:
                continue
            if normalized == candidate or normalized.startswith(f"{candidate}/"):#精确匹配或文件路径前缀匹配
                return True
        return False

    # ---------- 上下文窗口（父子 chunk 检索核心） ----------

    def _build_context_window(self, hit_doc: dict[str, Any]) -> str:
        """以命中的子 chunk 为中心，向同 parent 的兄弟 chunk 扩展，构建上下文。

        解决的问题：
        - 返回单个子 chunk（~1200 字符）→ 信息被拦腰切断
        - 返回整个 parent_text（最大 2632 字符）→ 太长，噪声多
        - 本方法折中：从命中点向左右扩展，累计到 MAX_CONTEXT_CHARS 就停

        manifest 实际数据：P75=595, P95=1357, max=2632。
        MAX_CONTEXT_CHARS=1500 覆盖了 95% 的完整 parent，只有极少数超长 section 需要窗口化。
        """
        parent_id = hit_doc.get("parent_id", "")
        # 找到同 parent 的所有兄弟 chunk（按 doc_id 自然排序）
        siblings = [
            d for d in self._documents if d.get("parent_id") == parent_id
        ]
        if len(siblings) <= 1:
            return str(hit_doc.get("text", ""))
        # 按 doc_id 排序确保文本顺序正确
        siblings.sort(key=lambda d: str(d.get("doc_id", "")))
        # 定位命中 chunk 的位置
        hit_doc_id = hit_doc.get("doc_id", "")
        hit_idx = next(
            (i for i, d in enumerate(siblings) if d.get("doc_id") == hit_doc_id),
            0,
        )
        # 从命中点向两边扩展
        left = hit_idx
        right = hit_idx
        total = len(str(siblings[hit_idx].get("text", "")))
        while total < MAX_CONTEXT_CHARS:
            expanded = False
            # 先向左扩一段
            if left > 0:
                left -= 1
                total += len(str(siblings[left].get("text", "")))
                expanded = True
                if total >= MAX_CONTEXT_CHARS:
                    break
            # 再向右扩一段
            if right < len(siblings) - 1:
                right += 1
                total += len(str(siblings[right].get("text", "")))
                expanded = True
                if total >= MAX_CONTEXT_CHARS:
                    break
            if not expanded:  # 左右都到边界了
                break
        parts = [
            str(siblings[i].get("text", "")) for i in range(left, right + 1)
        ]
        return "\n\n".join(parts).strip()

    # ---------- 检索 ----------

    def retrieve_vector(
        self,
        query: str,
        *,
        top_k: int = 4,
        path_filters: list[str] | None = None,
    ) -> list[Evidence]:
        self._ensure_loaded()
        if self._vector_index is None:
            return []

        retriever = self._vector_index.as_retriever(similarity_top_k=max(top_k * 4, top_k))#创建检索器，先取 top_k×4 个结果。后面要过滤路径，会丢掉一部分，多取保证过滤后还够。
        try:
            results = retriever.retrieve(query)
        except Exception:
            return []

        payload: list[Evidence] = []
        seen_parents: set[str] = set()  # 按 parent 去重
        for item in results:
            node = getattr(item, "node", item)
            metadata = getattr(node, "metadata", {}) or {}
            source_path = str(metadata.get("source_path", ""))
            if not self._matches_path_filters(source_path, path_filters):
                continue
            doc_id = str(metadata.get("doc_id", ""))
            raw_parent_id = metadata.get("parent_id")
            parent_id = str(raw_parent_id).strip() if raw_parent_id else None
            # 同一 parent 只保留最高分的一个结果
            if parent_id and parent_id in seen_parents:
                continue
            if parent_id:
                seen_parents.add(parent_id)
            # 找到 manifest 中对应的 chunk，用于构建上下文窗口
            hit_doc = next(
                (d for d in self._documents if d.get("doc_id") == doc_id), None
            )
            snippet = (
                self._build_context_window(hit_doc)
                if hit_doc
                else str(metadata.get("parent_text", ""))
            )
            payload.append(
                Evidence(
                    source_path=source_path,
                    source_type=str(metadata.get("source_type", "unknown")),
                    locator=str(metadata.get("locator", "")),
                    snippet=snippet,
                    channel="vector",
                    score=float(getattr(item, "score", 0.0) or 0.0),
                    parent_id=parent_id,
                )
            )
            if len(payload) >= top_k:
                break
        return payload

    def retrieve_bm25(
        self,
        query: str,
        *,
        top_k: int = 4,
        path_filters: list[str] | None = None, # 只搜指定目录
        query_hints: list[str] | None = None,#skill agent改写的query
    ) -> list[Evidence]:
        self._ensure_loaded()
        if not self._documents or not self._bm25_ready:
            return []

        hints = " ".join(query_hints or [])
        query_tokens = self._tokenize(f"{query} {hints}".strip())
        if not query_tokens:
            return []

        candidates = [
            item for item in self._documents if self._matches_path_filters(str(item["source_path"]), path_filters)#过滤出指定目录下的 chunk
        ]
        if not candidates:
            candidates = list(self._documents)

        scores: list[tuple[dict[str, Any], float]] = []
        corpus_size = max(1, len(self._documents))#文档（chunk）总数
        k1 = 1.5#词频饱和参数
        b = 0.75#长度惩罚参数
        for item in candidates:
            doc_tokens = item.get("tokens", [])# 文档的分词结果
            if not doc_tokens:
                continue
            token_counts = Counter(doc_tokens) # 文档内词频
            doc_len = len(doc_tokens)# 文档长度
            score = 0.0
            for token in query_tokens:# 对查询中的每个词
                if token not in token_counts:# 文档不包含该词 → 跳过
                    continue
                df = self._document_frequencies.get(token, 0) # 该词出现在多少文档中
                if df <= 0:
                    continue
                idf = math.log(1 + ((corpus_size - df + 0.5) / (df + 0.5)))#词越稀有越高
                freq = token_counts[token]
                denominator = freq + k1 * (1 - b + b * (doc_len / max(1.0, self._avg_doc_length)))
                score += idf * ((freq * (k1 + 1)) / max(denominator, 1e-9))
            if score > 0:
                scores.append((item, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        payload: list[Evidence] = []
        seen_parents: set[str] = set()
        for item, score in scores[:top_k]:
            raw_parent_id = item.get("parent_id")
            parent_id = str(raw_parent_id).strip() if raw_parent_id else None
            # 同一 parent 只保留最高分的一个结果
            if parent_id and parent_id in seen_parents:
                continue
            if parent_id:
                seen_parents.add(parent_id)
            snippet = self._build_context_window(item)
            payload.append(
                Evidence(
                    source_path=str(item["source_path"]),
                    source_type=str(item["source_type"]),
                    locator=str(item["locator"]),
                    snippet=snippet,
                    channel="bm25",
                    score=score,
                    parent_id=parent_id,
                )
            )
        return payload

    def _tokenize(self, text: str) -> list[str]:
        lowered = text.lower()
        tokens: list[str] = []
        tokens.extend(ALNUM_PATTERN.findall(lowered))#英文按单词切，数字也切出来。
        for match in CHINESE_BLOCK_PATTERN.findall(lowered):#对每个连续中文块，切两种粒度：
            tokens.extend(list(match))# 1. 每个汉字单独切成token
            if len(match) > 1:
                tokens.extend(match[index : index + 2] for index in range(len(match) - 1))# 2. 相邻两个汉字组合成一个token
        return tokens


knowledge_indexer = KnowledgeIndexer()