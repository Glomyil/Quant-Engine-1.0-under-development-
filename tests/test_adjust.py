# -*- coding: utf-8 -*-
"""
adjust.py 验收测试（脏数据矩阵版）

覆盖复权环节的**两条公式 + 三条约束 + 两类守卫**：

  ① 公式          hfq_close = close × factor ｜ hfq_open = open × factor（两列逐行相等）
  ①' 兄弟列        两列必须用各自的源列（open ≠ close，防止"把 close 抄进 hfq_open"）
  ② ★ 除权日不变量  除权当天：raw 收益 = −50%，hfq 收益 = 0.00（这是复权存在的唯一理由）
  ③ 缺失传播      close 缺 → hfq 缺；factor 缺 → hfq 缺（**绝不 fillna**）
  ④ 非法因子       factor = 0 / 负数 → 照样乘（不修复、不改值），只记账
  ⑤ 脏值转换      factor 写成字符串 / "abc" → to_numeric 转数值 / 变 NaN

  约束：只加两列（hfq_close / hfq_open）｜不改原列｜行数不变｜幂等
  守卫：空表 → ValueError；缺 back_adj_factor → ValueError；缺必需列 → ValueError

⚠️ 数据是故意造脏的：含除权日、0 因子、负因子、缺价、缺因子、字符串因子。
   —— 干净数据下"非法因子/缺失传播"那几行永远不会被真正执行（对应 PITFALLS #8/#10）。

用法（项目根目录）:
  python "tests/test_adjust.py"
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

from data.cleaner.adjust import run
from data.cleaner import base

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
# 测试数据：8 行，把复权能遇到的边界都放进去
#   行号 | 意图                | open  | close | factor | hfq_close | hfq_open
#    0   | 正常                |  9.90 | 10.00 |  2.0   | 20.0      | 19.80
#    1   | 正常                | 10.40 | 10.50 |  2.0   | 21.0      | 20.80
#    2   | ★ 除权日（10 送 10）|  5.20 |  5.25 |  4.0   | 21.0      | 20.80   ← raw −50%，hfq 0%
#    3   | 正常                |  5.35 |  5.40 |  4.0   | 21.6      | 21.40
#    4   | 非法因子 0          |  5.30 |  5.40 |  0.0   | 0.0       | 0.0
#    5   | open/close 缺失     |  NaN  |  NaN  |  4.0   | NaN       | NaN
#    6   | factor 缺失         |  5.45 |  5.50 |  NaN   | NaN       | NaN
#    7   | 非法因子 负数       |  5.55 |  5.60 | -1.0   | -5.6      | -5.55
#   ⚠️ open 故意 ≠ close：否则"把 close 当 open 用"的 bug 测不出来
# ---------------------------------------------------------------------------
def make_df():
    df = pd.DataFrame({
        "date":   pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07",
                                  "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-13"]),
        "open":   [ 9.90, 10.40, 5.20, 5.35, 5.30, np.nan, 5.45, 5.55],
        "high":   [10.2, 10.6, 5.30, 5.45, 5.45, np.nan, 5.55, 5.65],
        "low":    [ 9.8, 10.4, 5.20, 5.35, 5.35, np.nan, 5.45, 5.55],
        "close":  [10.0, 10.5, 5.25, 5.40, 5.40, np.nan, 5.50, 5.60],
        "volume": [100, 120, 200, 130, 110, 0, 140, 150],
        "amount": [1000, 1260, 1050, 702, 594, 0, 770, 840],
        "back_adj_factor": [2.0, 2.0, 4.0, 4.0, 0.0, 4.0, np.nan, -1.0],
        "code": "TEST",
    })
    return df


EXPECT_HFQ = [20.0, 21.0, 21.0, 21.6, 0.0, np.nan, np.nan, -5.6]
EXPECT_HFQ_OPEN = [19.8, 20.8, 20.8, 21.4, 0.0, np.nan, np.nan, -5.55]

DF = make_df()
OUT = run(DF, {}, CTX)

print("── ① 公式：hfq_close = close × factor ｜ hfq_open = open × factor ──")


def same(got, want):
    return all((np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9 for a, b in zip(got, want))


got_c = OUT["hfq_close"].to_numpy(dtype=float)
got_o = OUT["hfq_open"].to_numpy(dtype=float)
check(same(got_c, EXPECT_HFQ), f"hfq_close 逐行相等：{np.round(got_c, 4).tolist()}")
check(same(got_o, EXPECT_HFQ_OPEN), f"hfq_open  逐行相等：{np.round(got_o, 4).tolist()}")
check(str(OUT["hfq_close"].dtype) == "float64" and str(OUT["hfq_open"].dtype) == "float64",
      f"两列 dtype 都是 float64（实际 {OUT['hfq_close'].dtype} / {OUT['hfq_open'].dtype}）")
check(abs(got_c[0] - got_o[0]) > 1e-9, f"两列确实不同（20.0 vs 19.8）—— 用的是各自的源列")

print("\n── ② ★ 除权日不变量（复权存在的唯一理由）──")
raw_ret = DF["close"].pct_change(fill_method=None).iloc[2]
hfq_ret = OUT["hfq_close"].pct_change(fill_method=None).iloc[2]
check(abs(raw_ret - (-0.5)) < 1e-12, f"raw 口径当天收益 = {raw_ret:.2%}（10 送 10 造成的假暴跌）")
check(abs(hfq_ret - 0.0) < 1e-12, f"hfq 口径当天收益 = {hfq_ret:.2%}（正确：送股不产生收益）")
raw_o = DF["open"].pct_change(fill_method=None).iloc[2]
hfq_o = OUT["hfq_open"].pct_change(fill_method=None).iloc[2]
check(abs(raw_o - (-0.5)) < 1e-12, f"raw open 当天收益 = {raw_o:.2%}（同样被除权扭曲）")
check(abs(hfq_o - 0.0) < 1e-12, f"hfq open 当天收益 = {hfq_o:.2%}（开/收两列一起被抹平）")

print("\n── ③ 缺失传播：close 缺 / factor 缺 → hfq 缺（绝不 fillna）──")
check(np.isnan(OUT["hfq_close"].iloc[5]), "close = NaN → hfq_close = NaN")
check(np.isnan(OUT["hfq_close"].iloc[6]), "factor = NaN → hfq_close = NaN")
expect_nan_rows = int(DF["close"].isna().sum() + DF["back_adj_factor"].isna().sum())
check(int(OUT["hfq_close"].isna().sum()) == expect_nan_rows,
      f"NaN 行数 = {int(OUT['hfq_close'].isna().sum())}（close 缺 1 + factor 缺 1）")
D3 = make_df()
D3.loc[3, "open"] = np.nan                      # 只缺 open，close 完好
O3 = run(D3, {}, CTX)
check(np.isnan(O3["hfq_open"].iloc[3]) and O3["hfq_close"].iloc[3] == 21.6,
      "只缺 open：hfq_open 缺，hfq_close 照旧 21.6（两列互不污染）")
check(int(O3["hfq_open"].isna().sum()) == 3, f"此时 hfq_open NaN = {int(O3['hfq_open'].isna().sum())} 行（新缺 1 + 原有 2）")

print("\n── ④ 非法因子：0 / 负数 → 照样乘，不修复、不改值 ──")
check(OUT["hfq_close"].iloc[4] == 0.0, "factor = 0 → hfq_close = 0（不偷偷改成 NaN）")
check(OUT["hfq_close"].iloc[7] == -5.6, "factor = -1 → hfq_close = -5.6（不偷偷取绝对值）")
check(OUT["hfq_open"].iloc[4] == 0.0, "factor = 0 → hfq_open = 0（同样不偷偷改成 NaN）")
check(OUT["hfq_open"].iloc[7] == -5.55, "factor = -1 → hfq_open = -5.55（同样不取绝对值）")
check(OUT.attrs["adjust"]["factor_nonpositive_rows"] == 2,
      f"attrs 记账 factor_nonpositive_rows = {OUT.attrs['adjust']['factor_nonpositive_rows']}（0 和负数各 1）")

print("\n── ⑤ 脏值转换：factor 写成字符串 ──")
D2 = make_df()
D2["back_adj_factor"] = ["2.0", "2.0", "4.0", "4.0", "0.0", "4.0", "abc", "-1.0"]
O2 = run(D2, {}, CTX)
check(abs(O2["hfq_close"].iloc[0] - 20.0) < 1e-9, "字符串 '2.0' → 20.0（to_numeric 生效）")
check(np.isnan(O2["hfq_close"].iloc[6]), "字符串 'abc' → NaN（而不是报错或 0）")

print("\n── ⑥ 约束：只加两列 / 不改原列 / 行数不变 / 幂等 ──")
new_cols = [c for c in OUT.columns if c not in DF.columns]
check(new_cols == ["hfq_close", "hfq_open"], f"只新增两列：{new_cols}")
check(len(OUT) == len(DF), f"行数不变（{len(DF)} → {len(OUT)}）")
check(OUT["open"].equals(DF["open"]) and OUT["close"].equals(DF["close"])
      and OUT["back_adj_factor"].equals(DF["back_adj_factor"]),
      "open / close / back_adj_factor 一格未改")
check(OUT["date"].equals(DF["date"]), "date 列未被改动")
check(base.check_new_columns(list(DF.columns), list(OUT.columns)) == [], "新列都在白名单里（无未登记列）")
OUT2 = run(OUT, {}, CTX)
check(OUT2["hfq_close"].equals(OUT["hfq_close"]) and OUT2["hfq_open"].equals(OUT["hfq_open"]),
      "幂等：跑两遍两列都一致")

print("\n── ⑦ 报告：attrs 三个字段 ──")
check(set(OUT.attrs["adjust"]) == {"hfq_nan_rows", "hfq_open_nan_rows", "factor_nonpositive_rows"},
      f"字段集合 = {sorted(OUT.attrs['adjust'])}")
check(OUT.attrs["adjust"]["hfq_nan_rows"] == 2, f"hfq_nan_rows = {OUT.attrs['adjust']['hfq_nan_rows']}")
check(OUT.attrs["adjust"]["hfq_open_nan_rows"] == 2,
      f"hfq_open_nan_rows = {OUT.attrs['adjust']['hfq_open_nan_rows']}（open 缺 1 + factor 缺 1）")

print("\n── ⑧ 守卫：空表 / 缺 factor / 缺必需列 ──")
expect_raise(lambda: run(pd.DataFrame(), {}, CTX), "空表")
expect_raise(lambda: run(DF.drop(columns=["back_adj_factor"]), {}, CTX), "缺 back_adj_factor")
expect_raise(lambda: run(DF.drop(columns=["close"]), {}, CTX), "缺 close（必需列）")
expect_raise(lambda: run(DF.drop(columns=["open"]), {}, CTX), "缺 open（必需列）")

print("\n" + "=" * 46)
print(f"结果: 通过 {passed} / {passed + failed}")
if failed == 0:
    print("全部通过 ✅ adjust 复权环节可用")
else:
    print(f"仍有 {failed} 项失败 ❌ 请按上面提示修复")
    sys.exit(1)
