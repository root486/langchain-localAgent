<skills>
  <summary>Available local skills that the agent can inspect with read_file.</summary>
  <skill name="天气查询" path="skills/get_weather/SKILL.md">
    <description>查询指定城市的天气情况，并整理成适合直接回复用户的简洁结果。通过 tavily_web_search 工具直接搜索。</description>
  </skill>
  <skill name="失败恢复经验沉淀" path="skills/retry-lesson-capture/SKILL.md">
    <description>当一个任务首次执行失败，但在重试其他工具、接口、参数或流程后成功时，使用此技能把可复用经验总结进当前正在使用的 SKILL.md。适用于 API 失败后切换备用 API、命令失败后改用其他命令、抓取失败后改用其他数据源、解析失败后改用其他流程等场景。</description>
  </skill>
  <skill name="联网搜索" path="skills/web-search/SKILL.md">
    <description>使用 Tavily 联网搜索最新信息、官方文档、新闻动态、实时行情和外部事实来源。适用于用户明确要求搜索、联网、查官网、给链接、核验事实，或任务明显依赖实时外部信息的场景。通过 tavily_web_search / tavily_answer_search / tavily_news_search 工具直接调用。</description>
  </skill>
</skills>
