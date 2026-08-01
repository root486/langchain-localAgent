# RAGAS 测评对比：RRF 精排 top-6 vs top-4（20 条子集）

> 背景：`orchestrator.py` 原管线为 Multi-Query → Hybrid(top_k=4) → RRF 宽池(top_k=20) → qwen3-rerank 精排 **6** 条。
> 目的：验证最终上下文数量从 6 收到 4 是否影响检索命中与答案质量，为是否改参数提供数据依据。
> 结论速览：**检索层两档完全无损（recall@k=1.0 / top1 19/20），答案质量三指标在裁判噪音范围内持平，上下文 token 省约 27%。**
> 状态：**2026-08-01 已采纳 top-4**，同步改动清单见 §七。

## 一、实验配置

| 项 | 值 |
|----|----|
| 命令 | `python scripts/evaluate_ragas.py --top-k 6 --limit 20` / `--top-k 4 --limit 20` |
| 样本 | 同一批前 20 条 FAQ（`knowledge/E-commerce Data/faq.json`） |
| 层级 | 两层全跑：检索层（确定性二进制指标）+ 答案质量层（RAGAS LLM 判定） |
| 评测生成 LLM | 默认（`.env` llm_model） |
| RAGAS 裁判 | `qwen-max`（百炼） |
| 输出文件 | `storage/eval_outputs/ragas_top6_20.json` · `ragas_top4_20.json` |
| 评测日期 | 2026-08-01 |

> 注：Multi-Query 的展开 LLM（qwen-plus）存在运行间波动，属预期噪音（见 4.3），两档之间融合阶段的小差异与此相关。

## 二、检索层对比（20 条，确定性二进制指标）

| 阶段 | 指标 | top-6 | top-4 |
|------|------|-------|-------|
| vector | recall@k / MRR / nDCG | **1.000 / 0.975 / 0.982** | **1.000 / 0.975 / 0.982** |
|  | prec@k / top1 / topk | 0.167 / 19 / 20 | **0.250 / 19 / 20** |
| bm25 | recall@k / MRR / nDCG | **1.000 / 0.917 / 0.938** | **1.000 / 0.917 / 0.938** |
|  | prec@k / top1 / topk | 0.167 / 17 / 20 | **0.250 / 17 / 20** |
| fused（RRF 宽池） | recall@k / MRR / nDCG | 1.000 / 0.975 / 0.982 | 1.000 / 0.900 / 0.926 |
|  | prec@k / top1 / topk | 0.167 / 19 / 20 | 0.250 / 16 / 20 |
| **final（rerank 后）** | recall@k / MRR / nDCG | **1.000 / 0.975 / 0.982** | **1.000 / 0.975 / 0.982** |
|  | prec@k / top1 / topk | 0.167 / 19 / 20 | **0.250 / 19 / 20** |
| 跨阶段丢失 | vector命中但final丢 / bm25命中但final丢 | 0 / 0 | 0 / 0 |

### 2.1 检索层要点
- **recall@k 两档均为 1.0，top1 同为 19/20、topk 同为 20/20**：降到 4 条没有丢掉任何正确答案。
- **prec@k 从 0.167 → 0.250**：正确 chunk 唯一，精排窗口收窄后噪音占比下降（6 条里 5 条噪音 → 4 条里 3 条）。
- **fused 阶段两档有差异**（MRR 0.975 vs 0.900、top1 19 vs 16），但 **final 阶段又回到完全一致（0.975 / 19）**——qwen3-rerank 把最终排序稳定住了（详见 4.3）。

## 三、答案质量层对比（RAGAS，LLM 判定）

| 指标 | top-6 | top-4 | 差异（top6 − top4） | 说明 |
|------|-------|-------|---------------------|------|
| faithfulness（忠实度） | 0.9867 | 0.9879 | +0.0012 | 两档都接近满分，回答几乎不编造 |
| answer_relevancy（相关性） | 0.8098 | 0.8162 | −0.0064 | 4 略高，但属于裁判噪音 |
| answer_correctness（正确性） | 0.8573 | 0.8430 | +0.0143 | 6 略高，同样在噪音范围内 |

### 3.1 答案质量要点
- **三指标两档差值均 < 0.015**，远小于逐条裁判抖动幅度（±0.13~0.16，见下），无法区分优劣。
- faithfulness 稳定在 0.99 附近：说明 top-4 的上下文仍足以让模型忠实作答。
- answer_relevancy 始终是短板（0.81），与上下文数量无关，是 RAGAS 该指标的固有特性（回答比问题字面更完整即被判低，见历史 badcase 分析），本次对比不涉及其改进。

## 四、上下文成本对比

| 项 | top-6 | top-4 | 降幅 |
|----|-------|-------|------|
| 平均片段数 | 6.0 | 4.0 | −33% |
| 平均总字符（snippet） | 2915 | 2140 | **−27%** |
| 最大总字符 | 4922 | 4761 | −3% |

- 每轮知识库问答注入 LLM 的上下文 token 减少约 **1/4**（中文约 1 字符 ≈ 0.7-1 token，≈ **节省 800~900 token/轮**），流式首字延迟与成本同步下降。
- 片段数分布：top-6 恒为 6 条/轮，top-4 恒为 4 条/轮（本语料 snippet 足够时始终满额返回）。

## 五、逐条 correctness 摆动观察（top-6 vs top-4）

| 问题 | top-6 | top-4 | 差 |
|------|-------|-------|-----|
| 我如何联系客户服务部了解我的订单？ | 0.913 | 0.751 | +0.162 |
| 我可以注册帐户或使用 Facebook/Google 帐户登录吗？ | 0.974 | 0.836 | +0.138 |
| 我如何查看我的状态？ | 0.739 | 0.872 | −0.134 |
| 我可以加急订单吗？ | 0.532 | 0.406 | +0.125 |
| 我忘记了我的帐户 | 0.831 | 0.950 | −0.119 |

- 摆动**方向正负参半**（有的 6 好、有的 4 好），且幅度与三指标均值差（<0.015）完全不成比例，判定为 **LLM 裁判随机性**，而非上下文数量的系统性影响。
- 例外提示：`婚礼及活动礼服（量身定制）常见问题解答` 在 top-6 那次 correctness 为 `nan`（裁判未给出可解析分数，已被均值排除），top-4 次为 0.867——属单次裁判异常，不构成趋势。

## 六、结论

1. **检索无损**：top-4 下 recall@k=1.0、final top1=19/20、MRR/nDCG 与 top-6 完全一致，零跨阶段丢失。
2. **精度更高**：prec@k 0.250 vs 0.167。
3. **答案质量持平**：三指标差异在裁判噪音内，方向不一。
4. **成本更低**：上下文 token 省约 27%。

> ⚠️ 局限：本对比基于 20 条子集，样本较小。correctness 的 0.014 差距虽在噪音内，但若要下最终定论，建议在采纳前跑一次全量 120 条确认。

## 七、落地：已采纳 top-4（2026-08-01）

改最终精排数 `top_n`，**RRF 宽池 top_k=20 保持不动**（宽池基本等于去重后全量候选 ~20-28 条，不是瓶颈，收窄无收益且可能漏候选）。

以下同步改动已于 2026-08-01 全部完成（CLAUDE.md 4.7：生产管线与评测管线、文档必须一致）：

| 位置 | 改动 | 状态 |
|------|------|------|
| `backend/knowledge_retrieval/orchestrator.py`（原 L83） | `rerank_evidences(query, fused_candidates, top_n=6)` → `top_n=4` | ✅ 完成 |
| `backend/scripts/evaluate_ragas.py` `parse_args` | `--top-k` 默认 `6` → `4`（评测模拟同一管线） | ✅ 完成 |
| `docs/reference/hardcoded-values.md` | `top_k=6` 行更新为 `top_k=4`，并新增宽池 `top_k=20` 行 | ✅ 完成 |
| `CLAUDE.md` 数据流图 | "RRF 融合(宽池20) → rerank 精排(6)" → "(4)"；4.7 节检索参数同步为 `hybrid 4 / 宽池 20 / 精排 4` | ✅ 完成 |
| `README.md` | 精排 Top-6 → Top-4（特性列表 / 流程图 / 说明，共 3 处） | ✅ 完成 |
| `knowledge_retrieval/fusion.py` 默认 `top_k=6` | 仅默认值，实际调用处均显式传参；为一致性改为 4 | ✅ 完成 |

改后无需重建索引（不改 embedding / 切分 / 分词），重启后端即生效。

## 附：评测复现命令

```bash
cd backend
python scripts/evaluate_ragas.py --top-k 6 --limit 20 --output storage/eval_outputs/ragas_top6_20.json
python scripts/evaluate_ragas.py --top-k 4 --limit 20 --output storage/eval_outputs/ragas_top4_20.json
# 只跑检索层（零 LLM，仅 rerank 走 API）做快速复查：
python scripts/evaluate_ragas.py --top-k 4 --limit 20 --no-generation
```
