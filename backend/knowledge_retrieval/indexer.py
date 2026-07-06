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


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
ALNUM_PATTERN = re.compile(r"[A-Za-z0-9_]+")
CHINESE_BLOCK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


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
        sections: list[tuple[list[str], list[str]]] = []# 存结果：[(标题层级链, 内容行)]
        heading_stack: list[str] = []# 标题栈，跟踪当前在哪个标题层级下
        current_lines: list[str] = []# 当前标题下的内容行

        # 把当前积累的内容存为一个section
        def flush_section() -> None:
            if not current_lines:
                return
            heading_path = heading_stack[:] if heading_stack else [path.stem]
            sections.append((heading_path, current_lines[:]))
        #把 Markdown 文件按标题切成多块，每块知道自己在标题树的哪个位置。
        for raw_line in text.splitlines():
            match = HEADING_PATTERN.match(raw_line) # 匹配 # 标题
            if not match:
                current_lines.append(raw_line)# 非标题行，加入当前内容
                continue

            flush_section() # 遇到新标题，把之前的内容存起来
            current_lines = [raw_line]# ##标题，重置当前内容
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1]#先弹回到上级（[:level-1]）
            heading_stack.append(title)#再进入新标题（append）

        flush_section()
        # sections示例: [(["常见问题"], ["# 常见问题", ""]), (["常见问题", "退货"], ["## 退货", "退货政策..."]), (["常见问题", "退货", "退货流程"],        ["### 退货流程", "先联系客服，再寄回商品。"])]
        # 标题树:  常见问题
        #          ├── 退货
        #          │   ├── 退货流程
        #          │   └── 退货条件
        #          └── 换货
        if not sections:
            sections = [([path.stem], text.splitlines())]#如果文件没有任何标题，整个文件作为一块，用文件名当标签。

        chunks: list[dict[str, Any]] = []
        for section_index, (heading_path, lines) in enumerate(sections, start=1):
            section_text = "\n".join(lines).strip()#把内容行用换行符拼成一段文本。
            if not section_text:
                continue
            parent_id = f"{source_path}::{' > '.join(heading_path)}"#生成 parent_id，比如 "knowledge/faq.md::常见问题 > 退货"
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", section_text) if part.strip()]#按空行切开，切开后 → ["第一段内容", "第二段内容", "第三段内容"]
            if not paragraphs:
                paragraphs = [section_text]

            for paragraph_index, paragraph in enumerate(paragraphs, start=1):
                content = paragraph.strip()
                if not content:
                    continue
                slices = [content[index : index + 1200] for index in range(0, len(content), 1200)] or [content]#每1200字符切一块
                for slice_index, slice_text in enumerate(slices, start=1):
                    locator = f"{' > '.join(heading_path)} / 段落 {paragraph_index}"#生成位置描述。比如 "常见问题 > 退货 / 段落 1"
                    if len(slices) > 1:
                        locator = f"{locator}.{slice_index}"#如果超长被切成了多片，加序号区分："常见问题 > 退货 / 段落 1.1"
                    chunks.append(
                        {
                            "doc_id": f"{parent_id}::child::{paragraph_index}.{slice_index}",
                            "parent_id": parent_id,
                            "source_path": source_path,
                            "source_type": "md",
                            "locator": locator,
                            "text": slice_text,
                            "parent_text": section_text,
                            "section_index": section_index,
                        }
                    )
        return chunks

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
            locator = f"记录 {record_id}"
            parts = []
            if question:
                parts.append(f"Question: {question}")
            if answer:
                parts.append(f"Answer: {answer}")
            if label:
                parts.append(f"Label: {label}")
            if url:
                parts.append(f"URL: {url}")
            text = "\n".join(parts)
            parent_id = f"{source_path}::record::{record_id}"
            chunks.append(
                {
                    "doc_id": f"{parent_id}::child::1",
                    "parent_id": parent_id,
                    "source_path": source_path,
                    "source_type": "json",
                    "locator": locator,
                    "text": text,
                    "parent_text": text,
                    "record_id": record_id,
                }
            )
        return chunks
    #把所有 chunk 数据存到磁盘文件 manifest.json。
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
        except Exception:
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
        for item in results:
            node = getattr(item, "node", item)
            metadata = getattr(node, "metadata", {}) or {}
            source_path = str(metadata.get("source_path", ""))
            if not self._matches_path_filters(source_path, path_filters):
                continue
            text = getattr(node, "text", "") or getattr(node, "get_content", lambda: "")()
            raw_parent_id = metadata.get("parent_id")
            parent_id = str(raw_parent_id).strip() if raw_parent_id else None
            payload.append(
                Evidence(
                    source_path=source_path,
                    source_type=str(metadata.get("source_type", "unknown")),
                    locator=str(metadata.get("locator", "")),
                    snippet=str(text).strip(),
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
        for item, score in scores[:top_k]:
            raw_parent_id = item.get("parent_id")
            parent_id = str(raw_parent_id).strip() if raw_parent_id else None
            payload.append(
                Evidence(
                    source_path=str(item["source_path"]),
                    source_type=str(item["source_type"]),
                    locator=str(item["locator"]),
                    snippet=str(item["text"]).strip(),
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