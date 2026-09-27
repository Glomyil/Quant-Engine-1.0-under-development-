# -*- coding: utf-8 -*-
"""
tests/test_assemble.py —— assemble.py 验收（**装配行为**；需要 features + factors + assemble 三个文件）

覆盖规格 §2.3 装配规则：

  ① 只算 names 里因子 needs 的**并集**（用不到的指标一个不算）
  ② needs 里的指标名不在 FEATURES → 当场 KeyError
  ③ requires 的表不在 panels → 跳过该因子并**打印原因**（不抛异常）
  ④ 同一指标的两种参数（n=2 / n=3）在缓存里是两个 key，互不覆盖
  ⑤ 指标与面板同形状；panels 一格未被改动
  ➕ 同一指标被两个因子用 → **只算一次**（会话缓存命中）
  ➕ feats 的 key 约定 = f"{指标名}_{参数值}"（参数按名排序：ret_2 / ret_3）

（注册表本身 → tests/test_factors.py）
用法（项目根目录）: python tests/test_assemble.py
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

try:
    from backtest import assemble as AS
    from backtest import factors as FA
    from backtest import features as FE
except ImportError as e:
    print(f"❌ 还没写齐 features / factors / assemble（{e}）")
    print("   顺序：features.py → factors.py → assemble.py")
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


DAYS = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
CLOSE = pd.DataFrame({
    "sh.600000": [10.0, 11.0, 12.1],
    "sz.000001": [10.0, 20.0, 40.0],
    "sz.300001": [np.nan, 5.0, 5.0],
}, index=DAYS)


def make_panels():
    listed = CLOSE.notna().cummax() & CLOSE.notna()[::-1].cummax()[::-1]
    ret1 = CLOSE.ffill().pct_change(fill_method=None).where(listed)
    return {"hfq_close": CLOSE.astype("float32"), "hfq_open": CLOSE.astype("float32"),
            "ret_1d_hfq": ret1.astype("float32"), "can_trade": CLOSE.notna().astype(bool)}


P = make_panels()
TEMP = ("_t20", "_t3", "_t2b", "_bogus", "_amtdep")

print("── ① 只算 needs 的并集 ──")
FA.FACTORS["_t20"] = {"fn": lambda f, p: f["ret_2"], "sign": 1, "needs": [("ret", {"n": 2})]}
c1 = {}
o1 = AS.build_features(P, ["_t20"], c1)
check(set(o1) == {"ret_2"}, f"只产出被 needs 要的指标：{sorted(o1)}")
check(len(c1) == 1, f"缓存只 1 条（实际 {len(c1)}）")

print("\n── ② needs 里的指标名不在 FEATURES → 当场 KeyError ──")
FA.FACTORS["_bogus"] = {"fn": lambda f, p: f, "sign": 1, "needs": [("nope_thing", {"n": 2})]}
try:
    AS.build_features(P, ["_bogus"], {})
    check(False, "指标名不在 FEATURES → 竟然没报错")
except KeyError as e:
    check(True, f"当场 KeyError（{e}）")
except Exception as e:
    check(False, f"抛的是 {type(e).__name__}，不是 KeyError")

print("\n── ③ requires 的表不在 panels → 跳过 + 打印原因 ──")
FA.FACTORS["_amtdep"] = {"fn": lambda f, p: f["ret_2"], "sign": 1,
                         "needs": [("ret", {"n": 2})], "requires": ["amount"]}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    o3 = AS.build_features(P, ["_amtdep"], {})
check("_amtdep" not in o3 and buf.getvalue().strip() != "",
      f"被跳过且打印了原因：{buf.getvalue().strip()[:60]}")

print("\n── ④ 同一指标两种参数 = 两个 key ──")
FA.FACTORS["_t3"] = {"fn": lambda f, p: f["ret_3"], "sign": 1, "needs": [("ret", {"n": 3})]}
c4 = {}
o4 = AS.build_features(P, ["_t20", "_t3"], c4)
check(len(c4) == 2, f"缓存两条（实际 {len(c4)}）")
check(set(o4) == {"ret_2", "ret_3"}, f"key = 指标名_参数值（实际 {sorted(o4)}）")

print("\n── ⑤ 同形状 + 不改动 panels ──")
real = [n for n in FA.FACTORS if n not in TEMP]
before = {k: v.copy(deep=True) for k, v in P.items()}
c5 = {}
o5 = AS.build_features(P, real, c5)
bad = [k for k, v in o5.items() if v.shape != P["hfq_close"].shape]
check(not bad, f"所有指标与面板同形状（异常 {bad}）")
check(all(P[k].equals(before[k]) for k in P), "panels 四张表一格未改")

print("\n── ➕ 加测 ──")
FA.FACTORS["_t2b"] = {"fn": lambda f, p: f["ret_2"], "sign": -1, "needs": [("ret", {"n": 2})]}
c6 = {}
o6 = AS.build_features(P, ["_t20", "_t2b"], c6)
check(len(c6) == 1, f"两个因子要同一指标 → 只算一次（缓存 {len(c6)} 条）")
check(np.allclose(o6["ret_2"].to_numpy("float64"), FE.FEATURES["ret"](P, n=2).to_numpy("float64"),
                  atol=1e-6, equal_nan=True), "值等于现算的那一份（命中缓存）")
for k in TEMP:
    FA.FACTORS.pop(k, None)

print("\n" + "=" * 46)
print(f"结果: 通过 {passed} / {passed + failed}")
if failed:
    print(f"仍有 {failed} 项失败 ❌ 按上面提示修 assemble.py")
    sys.exit(1)
print("全部通过 ✅ 按需装配 + 会话缓存可用")
