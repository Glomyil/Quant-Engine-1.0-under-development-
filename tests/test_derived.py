# -*- coding: utf-8 -*-
"""
derived.py 验收测试（脏数据矩阵版）

覆盖派生环节的**一条公式 + 一个口径 + 三条约束 + 三类守卫**：

  ① 公式          ret_1d_hfq = hfq_close.ffill().pct_change(fill_method=None)
  ② ★ 停牌口径    停牌期间收益 = 0（不是 NaN）；**跳空集中在复牌日**
  ③ 除权不变量    除权日收益 = 0（复权在 adjust 已解决，派生环节不许再产生假暴跌）
  ④ 首行          首行收益 = NaN（没有"前一日"）

  约束：只加一列（ret_1d_hfq）｜不改原列｜行数不变｜幂等｜无 inf
  守卫：空表 → ValueError；缺 hfq_close → ValueError；缺必需列 → ValueError

⚠️ 数据是故意造脏的：含连续停牌两日 + 复牌跳空 + 除权日 + 缺价。
   —— 干净数据下"停牌=0 / 复牌跳空"这两个分支永远不会被真正执行（PITFALLS #8/#10）。

用法（项目根目录）:
  python "tests/test_derived.py"
期望末尾: 全部通过 (X/Y)
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.cleaner.derived import run

CTX = {"universe": "stock", "asset_type": "stock", "code": "TEST"}

passed = 0
failed = 0


def check(cond, label):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✅ {label}")
    else:
        failed += 1
        print(f"  ❌ {label}")


def expect_raise(fn, label):
    global passed, failed
    try:
        fn()
    except ValueError as e:
        passed += 1
        print(f"  ✅ {label}（{e}）")
    except Exception as e:                      # noqa: BLE001
        failed += 1
        print(f"  ❌ {label}：抛的是 {type(e).__name__}，不是 ValueError（{e}）")
    else:
        failed += 1
        print(f"  ❌ {label}：竟然没报错")


# ---------------------------------------------------------------------------
# 测试数据：7 行，覆盖"正常 / 连续停牌 / 复牌跳空 / 除权日"
#   行 | date   | close  | factor | hfq_close | 期望 ret_1d_hfq | 意图
#   0  | 01-02  | 10.00  |  2.0   | 20.00     | NaN             | 首行
#   1  | 01-03  | 10.50  |  2.0   | 21.00     | +5.00%          | 正常
#   2  | 01-06  | NaN    |  2.0   | NaN→ffill | 0.00%           | 停牌第 1 天
#   3  | 01-07  | NaN    |  2.0   | NaN→ffill | 0.00%           | 停牌第 2 天
#   4  | 01-08  | 11.55  |  2.0   | 23.10     | +10.00%         | ★ 复牌：累计跳空
#   5  | 01-09  |  5.775 |  4.0   | 23.10     | 0.00%           | ★ 除权日（10 送 10）
#   6  | 01-10  |  5.8905|  4.0   | 23.562    | +2.00%          | 正常
# ---------------------------------------------------------------------------
def make_df():
    return pd.DataFrame({
        "date":   pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07",
                                  "2020-01-08", "2020-01-09", "2020-01-10"]),
        "open":   [10.0, 10.5, np.nan, np.nan, 11.55, 5.775, 5.8905],
        "high":   [10.2, 10.6, np.nan, np.nan, 11.60, 5.800, 5.9000],
        "low":    [ 9.8, 10.4, np.nan, np.nan, 11.50, 5.700, 5.8000],
        "close":  [10.0, 10.5, np.nan, np.nan, 11.55, 5.775, 5.8905],
        "volume": [100, 120, 0, 0, 150, 200, 180],
        "amount": [1000, 1260, 0, 0, 1732, 1155, 1060],
        "back_adj_factor": [2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0],
        "hfq_close": [20.0, 21.0, np.nan, np.nan, 23.1, 23.1, 23.562],
        "code": "TEST",
    })


EXPECT_RET = [np.nan, 0.05, 0.0, 0.0, 0.10, 0.0, 0.02]

print("── ① 公式与口径 ──")
DF = make_df()
try:
    OUT = run(DF, {}, CTX)
except Exception as e:                          # noqa: BLE001
    print(f"  ❌ 环节直接崩溃：{type(e).__name__}: {e}")
    print("     （先修好 run() 第一行：out = df.copy() 要有括号）")
    sys.exit(1)

got = OUT["ret_1d_hfq"].to_numpy(dtype=float)
ok = all((np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9 for a, b in zip(got, EXPECT_RET))
check(ok, f"逐行与手算一致：{np.round(got, 4).tolist()}")
check(str(OUT["ret_1d_hfq"].dtype) == "float64", f"dtype = float64（实际 {OUT['ret_1d_hfq'].dtype}）")

print("\n── ② ★ 停牌口径：停牌期 = 0，跳空集中在复牌日 ──")
check(np.isnan(got[0]), "首行 = NaN（没有前一日）")
check(got[2] == 0.0 and got[3] == 0.0, f"连续停牌两日 = 0（{got[2]}, {got[3]}），不是 NaN")
check(abs(got[4] - 0.10) < 1e-9, f"复牌日 = +10.00%（累计跳空，实际 {got[4]:.4%}）")

print("\n── ③ 除权不变量：派生环节不许再产生假暴跌 ──")
raw_ret = DF["close"].pct_change(fill_method=None).iloc[5]
check(abs(raw_ret - (-0.5)) < 1e-12, f"raw 口径除权日 = {raw_ret:.2%}（假暴跌）")
check(got[5] == 0.0, f"ret_1d_hfq 除权日 = {got[5]:.2%}（正确：送股不产生收益）")

print("\n── ④ inf / 缺失传播 ──")
check(int(np.isinf(got).sum()) == 0, "全列无 inf")
check(int(np.isnan(got).sum()) == 1, f"NaN 行数 = {int(np.isnan(got).sum())}（只有首行）")

print("\n── ⑤ 约束：只加一列 / 不改原列 / 行数不变 / 幂等 ──")
new_cols = [c for c in OUT.columns if c not in DF.columns]
check(new_cols == ["ret_1d_hfq"], f"只新增一列：{new_cols}")
check(len(OUT) == len(DF), f"行数不变（{len(DF)} → {len(OUT)}）")
check(OUT["hfq_close"].equals(DF["hfq_close"]), "hfq_close 一格未改（ffill 只是中间步骤）")
check(OUT["close"].equals(DF["close"]), "close 一格未改")
OUT2 = run(OUT, {}, CTX)
check(OUT2["ret_1d_hfq"].equals(OUT["ret_1d_hfq"]), "幂等：跑两遍结果一致")

print("\n── ⑥ 报告：attrs 两个字段 ──")
check(set(OUT.attrs["derived"]) == {"ret_nan_rows", "ret_inf_rows"},
      f"字段集合 = {sorted(OUT.attrs['derived'])}")
check(OUT.attrs["derived"] == {"ret_nan_rows": 1, "ret_inf_rows": 0},
      f"计数 = {OUT.attrs['derived']}")

print("\n── ⑦ 守卫：空表 / 缺 hfq_close / 缺必需列 ──")
expect_raise(lambda: run(pd.DataFrame(), {}, CTX), "空表")
expect_raise(lambda: run(DF.drop(columns=["hfq_close"]), {}, CTX), "缺 hfq_close")
expect_raise(lambda: run(DF.drop(columns=["volume"]), {}, CTX), "缺 volume（必需列）")

print("\n" + "=" * 46)
print(f"结果: 通过 {passed} / {passed + failed}")
if failed == 0:
    print("全部通过 ✅ derived 派生环节可用")
else:
    print(f"仍有 {failed} 项失败 ❌ 请按上面提示修复")
    sys.exit(1)
