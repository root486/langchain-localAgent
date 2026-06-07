"use client";

import { Database, FileSearch, Plus, Sparkles, Wrench } from "lucide-react";

import { useAppStore } from "@/lib/store";

export function Navbar() {
  const {
    createNewSession,
    ragMode,
    toggleRagMode,
    compressCurrentSession,
    renameCurrentSession,
    rebuildKnowledgeIndex,
    knowledgeIndexStatus,
    sessions,
    currentSessionId
  } = useAppStore();

  const currentTitle =
    sessions.find((session) => session.id === currentSessionId)?.title ?? "新会话";
  const isIndexBuilding = Boolean(knowledgeIndexStatus?.building);
  const knowledgeIndexLabel = isIndexBuilding ? "索引重建中" : "重建索引";
  const knowledgeIndexHint = isIndexBuilding
    ? "知识索引构建中"
    : knowledgeIndexStatus?.ready
      ? `知识索引已就绪 · ${knowledgeIndexStatus.indexed_files} 个文件`
      : "知识索引未就绪";

  return (
    <header className="panel flex items-center justify-between rounded-[30px] px-5 py-4">
      <div className="flex items-center gap-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[rgba(15,139,141,0.14)] text-ocean">
          <Sparkles size={20} />
        </div>
        <div>
          <p className="text-xs uppercase tracking-[0.32em] text-[var(--color-ink-soft)]">
            skill-rag
          </p>
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-semibold tracking-[-0.04em]">{currentTitle}</h1>
            <button
              className="rounded-full border border-[var(--color-line)] px-3 py-1 text-xs text-[var(--color-ink-soft)]"
              onClick={() => {
                const next = window.prompt("重命名当前会话", currentTitle);
                if (next) {
                  void renameCurrentSession(next);
                }
              }}
              type="button"
            >
              Rename
            </button>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-end gap-3">
        <button
          className="flex items-center gap-2 rounded-full border border-[var(--color-line)] bg-white/60 px-4 py-2 text-sm"
          onClick={() => void createNewSession()}
          type="button"
        >
          <Plus size={16} />
          新会话
        </button>
        <button
          className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm ${
            ragMode
              ? "bg-ocean text-white"
              : "border border-[var(--color-line)] bg-white/60 text-ink"
          }`}
          onClick={() => void toggleRagMode()}
          type="button"
        >
          <Database size={16} />
          {ragMode ? "RAG 已开" : "RAG 已关"}
        </button>
        <button
          className="flex items-center gap-2 rounded-full border border-[var(--color-line)] bg-white/60 px-4 py-2 text-sm"
          onClick={() => void compressCurrentSession()}
          type="button"
        >
          <Wrench size={16} />
          压缩
        </button>
        <button
          className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm ${
            isIndexBuilding
              ? "cursor-not-allowed bg-[rgba(15,139,141,0.12)] text-ocean"
              : "border border-[var(--color-line)] bg-white/60"
          }`}
          disabled={isIndexBuilding}
          onClick={() => void rebuildKnowledgeIndex()}
          type="button"
        >
          <FileSearch size={16} />
          {knowledgeIndexLabel}
        </button>
        <div className="hidden items-center gap-2 rounded-full bg-[rgba(212,106,74,0.12)] px-4 py-2 text-sm text-[var(--color-ember)] md:flex">
          <FileSearch size={16} />
          {knowledgeIndexHint}
        </div>
      </div>
    </header>
  );
}
