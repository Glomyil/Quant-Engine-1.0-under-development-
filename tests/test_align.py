import sys
from pathlib import Path

import pandas as pd

# ① 先让 Python 能找到 data 包（这样写不管从哪个目录运行都对）
# 本文件若放在 tests/ 下 → 上一级就是项目根
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ② 再 import（顺序不能反：Python 的 import 是运行时执行的语句，不是声明）
from data.cleaner.align import run

# 故意造脏: 字符串日期 + 乱序 + 一天两行
df = pd.DataFrame({
    "date":  ["2026-01-02", "2026-01-01", "2026-01-02", "bad-date"],
    "open":  [2.0, 1.0, 9.9, 5.0],
    "high":  [2.2, 1.2, 9.9, 5.0],
    "low":   [1.9, 0.9, 9.9, 5.0],
    "close": [2.1, 1.05, 9.9, 5.0],
    "volume": [200, 100, 999, 0],
})
out = run(df, {}, {"universe": "stock"})
print(out)
# 期望: 2 行(坏日期那行被删, 01-02 重复只留 first 的 open=2.0)
assert list(out["date"]) == pd.to_datetime(
    ["2026-01-01", "2026-01-02"]).tolist()
assert out["date"].dtype == "datetime64[ns]"
print("align 验收通过 ")
