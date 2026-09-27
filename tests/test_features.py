# -*- coding: utf-8 -*-
"""
tests/test_features.py —— features.py 验收（手工小面板 fixture）

覆盖《教学文档/最小因子库与装配规格.md》§4.1 的 6 条断言 + 3 条加测（标 ➕）：

  ① 形状与 panels 完全一致（index / columns 都不变）
  ② 前 n 行必须是 NaN（min_periods=n 生效）
  ③ 与手算逐格一致（含"停牌缺行被 ffill 补上"那一格）
  ④ 全表无 inf（ffill 之后不会算出 inf）
  ⑤ 单调变换不变性：对价格取 log 重算，秩相关 = 1.0000
  ⑥ dtype = float32
  ➕ ret(1) 与面板 ret_1d_hfq 逐格相等（NaN 位置也要一样）→ 口径钉死
  ➕ 退市后的"在册区间"之外必须是 NaN（不是 ffill 拉出来的假 0 收益）
  ➕ vol 只读面板的 ret_1d_hfq：把该列 ×10 → vol 也 ×10（若重算 pct_change 则不变）

依赖接口（规格 §2.1 / §5）：FEATURES = {"ret": fn(panels, n=..), "vol": ..., "ma_dev": ...}
用法（项目根目录）: python tests/test_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

try:
    from backtest import features as FE
except ImportError as e:
    print(f"❌ 还没有 backtest/features.py（{e}）")
    print("   先照《最小因子库与装配规格》§5 写 features.py，再跑本测试")
    sys.exit(1)

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


DAYS = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
# A 停牌缺行 ｜ B 干净 ｜ C 后两天才上市 ｜ D 第 3 天起退市（后两天长表里没有这一行）
CLOSE = pd.DataFrame({
    "sh.600000": [10.0, np.nan, 11.0, 12.1],
    "sz.000001": [10.0, 20.0, 40.0, 80.0],
    "sz.300001": [np.nan, np.nan, 5.0, 5.0],
    "sh.600005": [10.0, 12.0, np.nan, np.nan],
}, index=DAYS)


def make_panels():
    """手工面板：ret_1d_hfq 按**清洗层口径**造（每只股票只在自己的行上算，退市后不给 0）"""
    listed = CLOSE.notna().cummax() & CLOSE.notna()[::-1].cummax()[::-1]
    ret1 = CLOSE.ffill().pct_change(fill_method=None).where(listed)
    return {"hfq_close": CLOSE.astype("float32"),
            "hfq_open": CLOSE.astype("float32"),
            "ret_1d_hfq": ret1.astype("float32"),
            "can_trade": CLOSE.notna().astype(bool)}


print(f"fixture：{CLOSE.shape[0]} 天 × {CLOSE.shape[1]} 只（A 停牌缺行 / B 干净 / C 晚上市 / D 早退市）")
P = make_panels()
RET2 = FE.FEATURES["ret"](P, n=2)
VOL2 = FE.FEATURES["vol"](P, n=2)

print("\n── ① 形状与 panels 完全一致 ──")
check(RET2.shape == CLOSE.shape and RET2.index.equals(CLOSE.index) and RET2.columns.equals(CLOSE.columns),
      f"ret(2) 形状/索引/列 = {RET2.shape}")
check(VOL2.shape == CLOSE.shape and VOL2.index.equals(CLOSE.index) and VOL2.columns.equals(CLOSE.columns),
      f"vol(2) 形状/索引/列 = {VOL2.shape}")

print("\n── ② 前 n 行必须是 NaN（min_periods=n）──")
check(RET2.iloc[:2].isna().to_numpy().all(), "ret(2) 前 2 行全 NaN")
check(VOL2.iloc[:2].isna().to_numpy().all(), "vol(2) 前 2 行全 NaN")

print("\n── ③ 与手算逐格一致 ──")
check(np.isclose(RET2.loc[DAYS[2], "sh.600000"], 0.1, atol=1e-6),
      f"A 第 3 天 ret(2) = {RET2.loc[DAYS[2], 'sh.600000']:.6f}（停牌缺行先 ffill 成 10.0 → 11/10−1）")
check(np.isclose(RET2.loc[DAYS[3], "sh.600000"], 0.21, atol=1e-6),
      f"A 第 4 天 ret(2) = {RET2.loc[DAYS[3], 'sh.600000']:.6f}（12.1/10−1）")
check(np.isclose(RET2.loc[DAYS[3], "sz.000001"], 3.0, atol=1e-6),
      f"B 第 4 天 ret(2) = {RET2.loc[DAYS[3], 'sz.000001']:.6f}（80/20−1）")
check(np.isclose(VOL2.loc[DAYS[3], "sz.000001"], 0.0, atol=1e-6),
      f"B 第 4 天 vol(2) = {VOL2.loc[DAYS[3], 'sz.000001']:.6f}（日收益恒定 1.0 → std = 0）")

print("\n── ④ 全表无 inf ──")
check(int(np.isinf(RET2.to_numpy()).sum()) == 0 and int(np.isinf(VOL2.to_numpy()).sum()) == 0,
      "ret(2) / vol(2) 都没有 inf")

print("\n── ⑤ 单调变换不变性（np.log(p).diff(n) → 排名必须完全一致）──")
LOGRET = np.log(CLOSE.ffill()).diff(2)          # = log(1+r)：r 的**严格单调**变换，所以排名必须一模一样
same_rank = []
for c in CLOSE.columns:
    m = RET2[c].notna() & LOGRET[c].notna()
    if m.sum() >= 2:
        same_rank.append(bool(RET2[c][m].rank().equals(LOGRET[c][m].rank())))
check(bool(same_rank) and all(same_rank), f"各列排名完全一致 = {same_rank}")

print("\n── ⑥ dtype = float32 ──")
check(RET2.dtypes.eq("float32").all() and VOL2.dtypes.eq("float32").all(),
      f"ret(2) / vol(2) dtype = {sorted(set(map(str, RET2.dtypes)))}")

print("\n── ➕ 加测（规格 §4.1 之外、但约束口径的三条）──")
RET1 = FE.FEATURES["ret"](P, n=1)
bad = int((~np.isclose(RET1.to_numpy("float64"), P["ret_1d_hfq"].to_numpy("float64"),
                       atol=1e-6, equal_nan=True)).sum())
check(bad == 0, f"ret(1) 与面板 ret_1d_hfq 逐格相等（NaN 位置也一致；不符 {bad} 格）")

check(RET2.loc[DAYS[2]:, "sh.600005"].isna().all(),
      f"退市股在册区间外是 NaN，不是假 0 收益：{RET2['sh.600005'].tolist()}")

P2 = make_panels()
P2["ret_1d_hfq"] = (P2["ret_1d_hfq"] * 10).astype("float32")
VOL2B = FE.FEATURES["vol"](P2, n=2)
check(np.allclose(VOL2.to_numpy("float64") * 10, VOL2B.to_numpy("float64"), atol=1e-5, equal_nan=True),
      "vol 只读 ret_1d_hfq：该列 ×10 → vol 也 ×10（重算 pct_change 的话不会变）")

if "ma_dev" in FE.FEATURES:
    MD = FE.FEATURES["ma_dev"](P, n=2)
    check(np.isclose(MD.loc[DAYS[3], "sh.600000"], 12.1 / 11.55 - 1, atol=1e-6),
          f"ma_dev(2) A 第 4 天 = {MD.loc[DAYS[3], 'sh.600000']:.6f}（12.1 / mean(11,12.1) − 1）")
    check(MD.dtypes.eq("float32").all(), "ma_dev(2) dtype = float32")
else:
    print("  ⚠️ FEATURES 里没有 ma_dev（规格 §5 有它）—— 不算失败，但建议补上")
if "amihud" in FE.FEATURES:
    # amihud 要面板里的 amount 表（阶段 1 没有，会被 requires 跳过）→ 这里喂一张**假**成交额，只为把它跑通：
    # 冒烟块与上面的断言都碰不到它，所以它的 typo 会一直藏到阶段 3 —— 这条就是补这个口子。
    P3 = make_panels()
    P3["amount"] = (P["hfq_close"].abs().fillna(1.0) * 1e6).astype("float32")
    AM = FE.FEATURES["amihud"](P3, n=2)
    check(AM.shape == CLOSE.shape and AM.columns.equals(CLOSE.columns), f"amihud(2) 形状/列 = {AM.shape}")
    check(AM.dtypes.eq("float32").all(), "amihud(2) dtype = float32")
    check(int(np.isinf(AM.to_numpy()).sum()) == 0, "amihud(2) 无 inf")
else:
    print("  ⚠️ FEATURES 里没有 amihud（规格 §5 有它）—— 缺了不算失败")

print("\n" + "=" * 46)
print(f"结果: 通过 {passed} / {passed + failed}")
if failed:
    print(f"仍有 {failed} 项失败 ❌ 按上面提示修 features.py")
    sys.exit(1)
print("全部通过 ✅ features 口径可用")
