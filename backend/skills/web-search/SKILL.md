---
name: 联网搜索
description: 使用 Tavily 联网搜索最新信息、官方文档、新闻动态、实时行情和外部事实来源。适用于用户明确要求搜索、联网、查官网、给链接、核验事实，或任务明显依赖实时外部信息的场景。通过 tavily_web_search / tavily_answer_search / tavily_news_search 工具直接调用。
---

## 目标

用 Tavily Search API 返回可追溯的联网结果，并给出带来源链接和时间说明的中文结论。

## 可用工具

| 工具 | 用途 |
|------|------|
| `tavily_web_search` | 通用网页搜索，适合文档、产品信息、事实查询 |
| `tavily_answer_search` | 搜索并生成直接答案，适合需要明确结论的查询 |
| `tavily_news_search` | 新闻搜索，适合最新动态、近期事件 |

所有工具共用参数：`query`（必填）、`max_results`（默认 5）、`search_depth`（basic/advanced）、`include_domains`、`exclude_domains`。

## 必备前提

- 环境变量 `TAVILY_API_KEY` 必须存在。
- 如果工具调用返回鉴权失败、限流或 API 异常，要明确说明失败原因。
- 不要假装联网成功，也不要偷偷切换到其他搜索引擎。

## 查询策略

1. 先把用户问题压缩成 1 个主查询，尽量控制在 400 个字符以内。
2. 复杂问题拆成多个子查询，分别调用。
3. 按场景选择工具：
   - `tavily_web_search`：官网、文档、常规事实、产品信息
   - `tavily_answer_search`：需要直接答案的问题
   - `tavily_news_search`：最新消息、今日动态、近期事件
4. 默认用 `search_depth=basic`；只有在需要更高相关性时才改 `advanced`。
5. 对金融查询，优先把中文需求改写成英文 ticker：
   - 不好：`今天黄金价格`
   - 更好：`XAU USD price today`
6. 对官方信息，优先加 `include_domains` 过滤：
   - `include_domains: ["docs.langchain.com"]`
   - `include_domains: ["openai.com"]`
7. 如果用户要求来源可核验，优先选高分且权威的结果；必要时再用 `fetch_url` 直接抓取返回的 URL 正文。

## 执行步骤

1. 判断问题是否真的需要联网。
2. 选择合适的工具（web_search / answer_search / news_search）。
3. 调用工具，关注返回结果中的 `title`、`url`、`score`、`published_date`、`content`。
4. 如果结果足够明确，直接整理答案并给出来源。
5. 如果多个结果冲突，优先最新且更权威的来源。
6. 对"今天 / 最新 / 当前"类查询，回答中必须写明查询日期或来源发布日期。

## 结果筛选规则

- 不要只用第一条结果下结论。
- 优先使用官方文档、官方公告、政府/学校/标准组织、主流一手媒体。
- 如果多个来源冲突，优先最新且更权威的来源，并显式说明冲突。
- 如果结果质量低，换查询词或加域名过滤重试。

## 输出格式

推荐输出：

```md
结论：...

依据：
1. ...
2. ...

来源：
- 标题 1: URL
- 标题 2: URL

时间说明：
- 查询时间：YYYY-MM-DD
- 或来源发布日期：YYYY-MM-DD
```

## 特别约束

- 对明显会变化的信息，不要省略时间说明。
- 对高风险信息，不要只引用二手总结页。
- 对实时行情，优先使用 `tavily_news_search` 并给出市场或计价单位。
