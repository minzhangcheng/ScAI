# Marker databases (offline ORA priors)

由 `scripts/fetch_markers.py` 拉取，**不纳入版本管理**（见 .gitignore）。

| 源 | 用途 | 许可 | URL |
|---|---|---|---|
| PanglaoDB | 主 | CC BY-NC 4.0 | https://panglaodb.se/ |
| CellMarker 2.0 | 补稀有/亚型 | 见官网 | http://bio-bigdata.hrbmu.edu.cn/CellMarker/ |

覆盖 human / mouse。仅离线学术用途；商用需复核许可。
更新：重跑 `python scripts/fetch_markers.py`。文件首行 `# SOURCE_VERSION=...` 记录版本。

> CellMarker 2.0 为 `.xlsx`，需 `openpyxl`；若缺则仅 PanglaoDB 可用（补充源失败不阻断，仅 warning）。
