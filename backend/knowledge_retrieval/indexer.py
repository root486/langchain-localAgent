from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import chromadb
import jieba
from rank_bm25 import BM25Okapi
from langsmith import traceable

from config import get_settings
from embeddings_client import EmbeddingClient
from knowledge_retrieval.types import Evidence, IndexStatus

# 屏蔽 jieba 首次分词时的 "Building prefix dict" 启动日志
jieba.setLogLevel(60)


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
ALNUM_PATTERN = re.compile(r"[A-Za-z0-9_]+")
CHINESE_BLOCK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
# \u53e5\u5b50\u8fb9\u754c\uff1a\u4e2d\u6587\u6807\u70b9\u6216\u6362\u884c\u540e\u7684\u975e\u7a7a\u884c\u9996
SENT_BOUNDARY = re.compile(r"[\u3002\uff01\uff1f\uff1b\n](?=\S)")
# \u4e0a\u4e0b\u6587\u7a97\u53e3\u4e0a\u9650\uff08\u5b57\u7b26\u6570\uff09\uff1aP95 \u7684 parent_text \u4e3a 1357\uff0c\u53d6 1500 \u8986\u76d6\u7edd\u5927\u90e8\u5206\u573a\u666f
MAX_CONTEXT_CHARS = 1500
# \u6700\u5c0f chunk \u5927\u5c0f\uff1a\u4f4e\u4e8e\u6b64\u503c\u7684\u76f8\u90bb\u540c parent chunk \u4f1a\u88ab\u5408\u5e76
MIN_CHUNK_SIZE = 60
# BM25 参数（rank_bm25 BM25Okapi）
# b 调成 0.25：本语料 chunk 长度差异大（FAQ 短问题 + 长 answer），
# b=0.75 默认值会过度惩罚长 chunk，导致精确匹配的 FAQ 记录被压制。
# 实测 FAQ top-1 命中率：b=0.75 → 89.2%，b=0.25 → 95.0%（旧手写实现 93.3%）。
BM25_K1 = 1.5
BM25_B = 0.25
# ChromaDB 知识库 collection 名
VECTOR_COLLECTION_NAME = "knowledge_chunks"


class KnowledgeIndexer:
    def __init__(self) -> None:
        self.base_dir: Path | None = None # 项目根目录
        self._chroma_client: Any | None = None# ChromaDB PersistentClient（懒加载）
        self._vector_collection: Any | None = None# ChromaDB 知识库 collection
        self._documents: list[dict[str, Any]] = []# 所有chunk的列表
        self._lock = threading.Lock()  # 线程锁（防止并发重建）
        self._building = False # 是否正在建索引
        self._last_built_at: float | None = None # 上次建索引时间
        self._bm25: BM25Okapi | None = None # rank_bm25 索引（configure/rebuild 时构建）
        self._vector_ready = False # 向量索引是否就绪
        self._bm25_ready = False # BM25索引是否就绪

    #configure 被调用后，如果磁盘上有之前建的索引，直接加载，不用重建。
    def configure(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True) # storage/knowledge/
        self._vector_dir.mkdir(parents=True, exist_ok=True) # storage/knowledge/vector/
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
    def _chroma_dir(self) -> Path:
        # ChromaDB 持久化目录（chroma.sqlite3 + index/），旧 LlamaIndex JSON 同目录并存
        return self._vector_dir / "chroma"

    def _supports_embeddings(self) -> bool:
        return bool(get_settings().embedding_api_key)

    def _get_chroma_client(self) -> chromadb.ClientAPI:
        """懒加载 ChromaDB 嵌入式客户端（本地 SQLite，无需服务端）。"""
        if self._chroma_client is None:
            from chromadb.config import Settings as ChromaSettings

            self._chroma_client = chromadb.PersistentClient(
                path=str(self._chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._chroma_client

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

    def source_paths(self, max_entries: int = 40) -> list[str]:
        """返回去重、排序的源文件相对路径（供路由 prompt 展示索引覆盖范围）。"""
        if not self._documents:
            return []
        paths = sorted({str(item.get("source_path", "")) for item in self._documents})
        return paths[:max_entries]
    #建索引的主要方法，先读文件切分成chunk，然后持久化到manifest.json，再统计BM25数据，最后建向量索引
    def rebuild_index(self) -> None:
        if self.base_dir is None:
            return

        with self._lock: # 加锁，防止并发重建
            self._building = True
            try:
                self._documents = self._build_documents()# 读文件+切分
                self._write_manifest()  # 持久化到manifest.json
                self._build_bm25_index()# 构建 BM25 索引（rank_bm25）
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
            elif suffix == ".pdf":
                documents.extend(self._split_pdf(path))# pdf文件按页切
            # .xlsx/.xls 不进索引：表格行数据对 RAG 召回价值低，且原 Skill Agent 已移除
            # （如需索引，补充 pandas + openpyxl 依赖并实现 _split_excel）
            # .txt 不进索引：历史提取残留，既有 PDF 同文重复、也有独立内容，用户决定暂不纳入
            # （如需索引，复用 _split_markdown 的句子边界切分 + 有界 parent_text 即可）
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

    def _split_pdf(self, path: Path) -> list[dict[str, Any]]:
        """PDF 按页切分：每页一个 parent，长页按句子边界拆子 chunk。

        文本提取用 PyMuPDF（fitz），对中文报告 PDF（CID 字体）支持最好；
        提取不到文本的页（如图片封面/扫描页）直接跳过，不崩溃。
        """
        try:
            import fitz  # 延迟导入：仅在有 PDF 文件时加载
        except ImportError:
            print(f"[索引] 未安装 pymupdf，跳过 PDF: {path.name}")
            return []

        source_path = self._relative_path(path)
        chunks: list[dict[str, Any]] = []
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            print(f"[索引] 打开 PDF 失败，跳过: {path.name} ({exc})")
            return chunks

        try:
            for page_index, page in enumerate(doc, start=1):
                try:
                    page_text = (page.get_text() or "").strip()
                except Exception:
                    continue
                if not page_text:
                    continue
                parent_id = f"{source_path}::page::{page_index}"
                slices = self._split_long_text(page_text)
                for si, slice_text in enumerate(slices, start=1):
                    locator = f"第 {page_index} 页"
                    if len(slices) > 1:
                        locator = f"{locator} (片段 {si}/{len(slices)})"
                    chunks.append(
                        {
                            "parent_id": parent_id,
                            "source_path": source_path,
                            "source_type": "pdf",
                            "locator": locator,
                            "text": slice_text,
                            "parent_text": page_text,
                        }
                    )
        finally:
            doc.close()
        # 同页相邻过短片段合并 + 分配 doc_id，与 markdown 路径一致
        return self._merge_short_chunks(chunks, source_path)

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
            self._bm25 = None
            self._bm25_ready = False
            return
        try:
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._documents = []
            self._bm25 = None
            self._bm25_ready = False
            return
        self._documents = list(payload.get("documents", []))
        self._last_built_at = payload.get("built_at")
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """用 rank_bm25 构建 Okapi BM25 索引。

        分词：jieba（中文）+ 字母数字正则（英文/数字），k1=1.5, b=0.25（见 BM25_K1/BM25_B）。
        BM25 索引在加载 manifest 时重建，token 不持久化到 manifest。
        """
        if not self._documents:
            self._bm25 = None
            self._bm25_ready = False
            return

        corpus = [
            self._tokenize(str(item.get("text", "")))
            for item in self._documents
        ]
        try:
            self._bm25 = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
            self._bm25_ready = True
        except Exception as exc:
            print(f"[索引] BM25 索引构建失败: {exc}")
            self._bm25 = None
            self._bm25_ready = False

    def _build_vector_index(self) -> None:
        if not self._supports_embeddings() or not self._documents:
            self._vector_collection = None
            self._vector_ready = False
            return

        try:
            client = self._get_chroma_client()
            # 重建：先删旧 collection（含旧 LlamaIndex 遗留数据，格式不兼容）
            try:
                client.delete_collection(VECTOR_COLLECTION_NAME)
            except Exception:
                pass
            collection = client.get_or_create_collection(
                name=VECTOR_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[索引] 正在向量化 {len(self._documents)} 个 chunk（批大小=10）...")
            texts = [str(item["text"]) for item in self._documents]
            embeddings = EmbeddingClient().embed(texts)
            collection.add(
                # id 仅需唯一；真实 doc_id 存在 metadata 里，供检索时回查 manifest
                ids=[str(i) for i in range(len(self._documents))],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {
                        "doc_id": str(item["doc_id"]),
                        "parent_id": str(item.get("parent_id") or ""),
                        "source_path": str(item["source_path"]),
                        "source_type": str(item["source_type"]),
                        "locator": str(item["locator"]),
                    }
                    for item in self._documents
                ],
            )
            self._vector_collection = collection
            self._vector_ready = True
            print(f"[索引] 向量化完成，已持久化到 {self._chroma_dir}")
        except Exception as exc:
            print(f"[索引] 向量化失败: {exc}")
            import traceback
            traceback.print_exc()
            self._vector_collection = None
            self._vector_ready = False

    def _load_vector_index(self) -> None:
        if not self._supports_embeddings():
            self._vector_collection = None
            self._vector_ready = False
            return
        # 没有 ChromaDB 持久化数据（chroma.sqlite3）→ 无历史向量索引，跳过。
        if not (self._chroma_dir / "chroma.sqlite3").exists():
            self._vector_collection = None
            self._vector_ready = False
            return
        try:
            client = self._get_chroma_client()
            collection = client.get_or_create_collection(
                name=VECTOR_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            if collection.count() == 0:
                self._vector_collection = None
                self._vector_ready = False
                return
            self._vector_collection = collection
            self._vector_ready = True
        except Exception:
            self._vector_collection = None
            self._vector_ready = False

    def _ensure_loaded(self) -> None:
        if not self._documents:
            self._load_manifest()
        if self._vector_collection is None and self._supports_embeddings():
            self._load_vector_index()
    #判断“这个 chunk 是否匹配指定的 path_filters（目录前缀）”
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

    @traceable(run_type="retriever", name="vector_retrieve")
    def retrieve_vector(
        self,
        query: str,
        *,
        top_k: int = 4,
        path_filters: list[str] | None = None,
    ) -> list[Evidence]:
        self._ensure_loaded()
        if self._vector_collection is None:
            return []

        try:
            query_embedding = EmbeddingClient().embed_one(query)
            results = self._vector_collection.query(
                query_embeddings=[query_embedding],
                # 先取 top_k×4 个结果。后面要过滤路径，会丢掉一部分，多取保证过滤后还够。
                n_results=max(top_k * 4, top_k),
                include=["metadatas", "distances"],
            )
        except Exception:
            return []

        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        if not ids or not metadatas:
            return []

        payload: list[Evidence] = []
        seen_parents: set[str] = set()  # 按 parent 去重
        for i, metadata in enumerate(metadatas):
            metadata = metadata or {}
            source_path = str(metadata.get("source_path", ""))
            if not self._matches_path_filters(source_path, path_filters):
                continue
            doc_id = str(metadata.get("doc_id", ""))
            raw_parent_id = metadata.get("parent_id")
            parent_id = str(raw_parent_id).strip() if raw_parent_id else None
            # 同一 parent 只保留最高分的一个结果（ChromaDB 按距离升序=相似度降序返回，首个即最高分）
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
            # ChromaDB cosine 距离 → 相似度（越高越好），语义与旧 LlamaIndex 一致
            distance = float(distances[i]) if i < len(distances) else 0.0
            payload.append(
                Evidence(
                    source_path=source_path,
                    source_type=str(metadata.get("source_type", "unknown")),
                    locator=str(metadata.get("locator", "")),
                    snippet=snippet,
                    channel="vector",
                    score=1.0 - distance,
                    parent_id=parent_id,
                )
            )
            if len(payload) >= top_k:
                break
        return payload

    @traceable(run_type="retriever", name="bm25_retrieve")
    def retrieve_bm25(
        self,
        query: str,
        *,
        top_k: int = 4,
        path_filters: list[str] | None = None, # 只搜指定目录
        query_hints: list[str] | None = None,#BM25 附加的检索提示词
    ) -> list[Evidence]:
        self._ensure_loaded()
        if not self._documents or not self._bm25_ready or self._bm25 is None:
            return []

        hints = " ".join(query_hints or [])
        query_tokens = self._tokenize(f"{query} {hints}".strip())
        # rank_bm25 的 get_scores 会按 query 词频重复计权，去重避免同一词被重复加分
        query_tokens = list(dict.fromkeys(query_tokens))
        if not query_tokens:
            return []

        # BM25Okapi.get_scores 返回与 self._documents 顺序一一对应的分数数组
        scores = self._bm25.get_scores(query_tokens)
        ranked: list[tuple[dict[str, Any], float]] = []
        for i, item in enumerate(self._documents):
            if not self._matches_path_filters(str(item["source_path"]), path_filters):
                continue
            score = float(scores[i])
            if score > 0:
                ranked.append((item, score))
        # 与旧逻辑一致：path 过滤后无结果则退回全库
        if not ranked:
            ranked = [
                (item, float(scores[i]))
                for i, item in enumerate(self._documents)
                if scores[i] > 0
            ]

        ranked.sort(key=lambda pair: pair[1], reverse=True)
        payload: list[Evidence] = []
        seen_parents: set[str] = set()
        for item, score in ranked[:top_k]:
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
        """分词：英文/数字按词切，中文用 jieba 精确模式。

        与旧手写 unigram+bigram 不同，jieba 产出真实词，符合标准 BM25 用法。
        保留重复 token（文档词频是 BM25 的输入）；查询端去重在 retrieve_bm25 处理。
        """
        lowered = text.lower()
        tokens: list[str] = ALNUM_PATTERN.findall(lowered)
        for block in CHINESE_BLOCK_PATTERN.findall(lowered):
            tokens.extend(w for w in jieba.lcut(block) if w.strip())
        return tokens


knowledge_indexer = KnowledgeIndexer()