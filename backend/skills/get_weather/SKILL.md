---
name: 天气查询
description: 查询指定城市的天气情况，并整理成适合直接回复用户的简洁结果。通过高德地图 MCP 的 maps_weather 工具直接查询。
---

## 执行步骤

1. 调用 `maps_weather`，参数使用 `{"city": "城市名"}` 格式，例如 `{"city": "深圳"}`
2. 从返回结果的 `lives` 数组（实时天气）中提取字段：
   - `weather`：天气状况
   - `temperature`：实时温度（℃）
   - `winddirection`：风向
   - `windpower`：风力
   - `humidity`：湿度
   - `reporttime`：数据发布时间
3. 用中文整理成简明结果回复用户，并注明数据来源为高德天气及发布时间

## 注意事项

- 若 `maps_weather` 调用失败（如城市名无法解析），先调用 `maps_geo` 解析该城市的 adcode，再以 adcode 作为 `city` 重试
- 用户只给模糊地点（如"深圳今天天气"）时，直接按用户给出的城市名查询即可，不要自行猜测其他城市

## 输出格式

用中文给出简明结果，包括：
- 温度
- 天气状况
- 风向/风力
- 湿度（可选）
- 数据来源和时间（取自 `reporttime`）
