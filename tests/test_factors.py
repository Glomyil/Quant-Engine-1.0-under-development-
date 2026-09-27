# -*- coding: utf-8 -*-
"""
tests/test_factors.py —— factors.py 验收（**只验注册表**，不需要 assemble.py）

覆盖规格 §4.2 的注册表部分：

  ① 每个因子都有 fn / sign / needs，且 sign ∈ {+1, −1}
  ② needs 里的指标名都在 FEATURES 里（拼错就当场报错，别等装配时才炸）
  ③ fn(feats, {}) 返回与面板同形状的 DataFrame
  ④ fn 不改动 feats（快照对比）
  ⑤ requires 里写的不是指标名（requires = 面板表，needs = 指标）

（装配行为 → tests/test_assemble.py）
用法（项目根目录）: python tests/test_factors.py
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
    from backtest import factors as FA
    from backtest import features as FE
except ImportError as e:
    print(f"❌ 还没写齐 backtest/factors.py 与 backtest/features.py（{e}）")
    print("   先照《factors.py 教学参考》§7 写 factors.py，再跑本测试")
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
CLOSE = pd.DataFrame({
    "sh.600000": [10.0, np.nan, 11.0, 12.1],
    "sz.000001": [10.0, 20.0, 40.0, 80.0],
    "sz.300001": [np.nan, np.nan, 5.0, 5.0],
}, index=DAYS)


def make_panels():
    listed = CLOSE.notna().cummax() & CLOSE.notna()[::-1].cummax()[::-1]
    ret1 = CLOSE.ffill().pct_change(fill_method=None).where(listed)
    return {"hfq_close": CLOSE.astype("float32"), "hfq_open": CLOSE.astype("float32"),
            "ret_1d_hfq": ret1.astype("float32"), "can_trade": CLOSE.notna().astype(bool)}


def feats_for(panels, spec):
    """按规格 §2.2 的 key 约定（指标名_参数值；参数按名排序）现造这个因子要的 feats"""
    out = {}
    for fname, kw in spec["needs"]:
        key = fname + "_" + "_".join(str(v) for _, v in sorted(kw.items()))
        out[key] = FE.FEATURES[fname](panels, **kw)
    return out


P = make_panels()
print(f"fixture：{CLOSE.shape[0]} 天 × {CLOSE.shape[1]} 只 ｜ 注册表里 {len(FA.FACTORS)} 个因子")
check(len(FA.FACTORS) > 0, f"注册表里至少有一个因子（现在 {len(FA.FACTORS)} 个）—— 空 dict 会让下面所有循环空转，全是假绿")

print("\n── ① 字段完整 + sign ∈ {+1,−1} ──")
for nm, spec in FA.FACTORS.items():
    ok = all(k in spec for k in ("fn", "sign", "needs")) and spec.get("sign") in (1, -1)
    check(ok, f"{nm}: fn/sign/needs 齐，sign = {spec.get('sign')!r}")

print("\n── ② needs 里的指标名都在 FEATURES 里 ──")
for nm, spec in FA.FACTORS.items():
    missing = [f for f, _ in spec["needs"] if f not in FE.FEATURES]
    check(not missing, f"{nm}: needs = {[f for f, _ in spec['needs']]}（缺 {missing}）")

print("\n── ③ fn 返回与面板同形状 ──")
for nm, spec in FA.FACTORS.items():
    if any(t not in P for t in spec.get("requires", [])):
        print(f"  ⏭ {nm}: requires={spec['requires']} 面板里没有 → 跳过（装配时也同样跳过）")
        continue
    out = spec["fn"](feats_for(P, spec), {})
    check(isinstance(out, pd.DataFrame) and out.shape == CLOSE.shape,
          f"{nm}: fn 返回 {type(out).__name__} {getattr(out, 'shape', None)}")

print("\n── ④ fn 不改动 feats ──")
for nm, spec in FA.FACTORS.items():
    if any(t not in P for t in spec.get("requires", [])):
        continue
    feats = feats_for(P, spec)
    before = {k: v.copy(deep=True) for k, v in feats.items()}
    spec["fn"](feats, {})
    check(all(feats[k].equals(before[k]) for k in feats), f"{nm}: 没动 feats（{len(feats)} 张）")

print("\n── ⑤ requires 里只能写面板表名（不能写指标名）──")
for nm, spec in FA.FACTORS.items():
    bad = [t for t in spec.get("requires", []) if t in FE.FEATURES]
    check(not bad, f"{nm}: requires = {spec.get('requires', [])}（混进指标名 {bad}）")

print("\n" + "=" * 46)
print(f"结果: 通过 {passed} / {passed + failed}")
if failed:
    print(f"仍有 {failed} 项失败 ❌ 按上面提示修 factors.py")
    sys.exit(1)
print("全部通过 ✅ 因子注册表可用（装配行为见 tests/test_assemble.py）")
