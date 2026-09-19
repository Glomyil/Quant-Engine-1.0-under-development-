# -*- coding: utf-8 -*-
"""
suspension.py 验收测试（脏数据矩阵版）

覆盖（对应 教学文档/suspension.py教学参考.md §4）：

  ① 主判据     tradestatus==0 的三种形态（僵尸 / 量空 / 全空）都打上 is_suspended=1
  ② 游程       suspension_days：连续第几天，复牌重置为 0
  ③ 约束       不删行 ｜ 不改原 df ｜ 只新增 is_suspended + suspension_days（顺序追加）
  ④ 记账       df.attrs["suspension"] 五个计数（僵尸 / 量空 / 全空 / 矛盾 / 停牌总数）
  ⑤ 幂等       run(run(df)) == run(df)
  ⑥ 守卫       空表 → ValueError ｜ 缺 tradestatus → ValueError ｜ 缺 amount 不应该报错
  ⑦ 边界       全停牌 / 全正常 / 开头就停牌 / 停-复-停 / 单行停牌 / 矛盾行
  ⑧ 真实数据   sh.600000（僵尸连停）· sh.600006（NaN 量）· sz.000022（全空行）
  ⑨ 扩展区     ← 在这里注入你自己的异常与特殊情况

⚠️ 数据是故意造脏的：四价相等 / volume=0 / volume=NaN / 四价全 NaN / 停牌却有量。
   —— 干净数据（全 tradestatus=1）测不出任何东西（PITFALLS #10）。

用法（项目根目录）:
  python "tests/test_suspension.py"            快速版（秒级）
  python "tests/test_suspension.py" --full     追加全库对账（约 1 分钟）
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from config.constants import STOCK_DAILY_DIR
from data.cleaner.suspension import run

CTX = {"universe": "stock", "asset_type": "stock", "code": "TEST"}
EXPECTED_LIBRARY_SUSPENDED = 282_893     # 体检 ops/health_check.py 的 C14 口径（2026-09-18 实测）
FULL = "--full" in sys.argv

passed = 0
failed = 0
skipped = 0


def check(cond, label):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [OK] {label}")
    else:
        failed += 1
        print(f"  [!!] {label}")


def expect_raise(fn, label, err=ValueError):
    global passed, failed
    try:
        fn()
    except err as e:
        passed += 1
        print(f"  [OK] {label}（{type(e).__name__}: {str(e)[:40]}）")
    except Exception as e:                      # noqa: BLE001
        failed += 1
        print(f"  [!!] {label}: 抛了 {type(e).__name__} 而非 {err.__name__}: {e}")
    else:
        failed += 1
        print(f"  [!!] {label}: 期望抛 {err.__name__}，但没有抛")


def expect_ok(fn, label):
    global passed, failed
    try:
        r = fn()
    except Exception as e:                      # noqa: BLE001
        failed += 1
        print(f"  [!!] {label}: 不该报错却抛了 {type(e).__name__}: {e}")
        return None
    passed += 1
    print(f"  [OK] {label}")
    return r


def make_df(tradestatus, volume=None, price=None, start="2026-01-05"):
    """快速造一份"外观可控"的 df（date 递增，四价统一取 price）。

    形态对照（本环节的四种外观）：
        正常行   tradestatus=1, price=10.0, volume=1000
        僵尸行   tradestatus=0, price=10.0, volume=0
        量空行   tradestatus=0, price=10.0, volume=nan
        全空行   tradestatus=0, price=nan,  volume=nan
        矛盾行   tradestatus=0, price=10.0, volume=500   <- 停牌却有成交量
    """
    n = len(tradestatus)
    volume = [1000] * n if volume is None else list(volume)
    price = [10.0] * n if price is None else list(price)

    def isna(x):
        return x != x                            # NaN 不等于自己（兼容 None）

    amount = [float("nan") if (isna(p) or isna(v)) else float(p) * float(v)
              for p, v in zip(price, volume)]
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="D"),
        "open": price, "high": price, "low": price, "close": price,
        "volume": volume, "amount": amount,
        "tradestatus": tradestatus,
    })


# ===========================================================================
# 测试数据：5 行，每行一种形态（与教学文档 §4 逐字一致）
#   行 0 正常 ｜ 行 1 僵尸 ｜ 行 2 量空 ｜ 行 3 全空 ｜ 行 4 正常
# ===========================================================================
DIRTY = pd.DataFrame({
    "date":   pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]),
    "open":   [10.0, 10.0, 10.0, float("nan"), 10.5],
    "high":   [10.5, 10.0, 10.0, float("nan"), 10.8],
    "low":    [9.8,  10.0, 10.0, float("nan"), 10.4],
    "close":  [10.2, 10.0, 10.0, float("nan"), 10.6],
    "volume": [1000, 0,    float("nan"), float("nan"), 1200],
    "amount": [10200.0, 0.0, float("nan"), float("nan"), 12700.0],
    "tradestatus": [1, 0, 0, 0, 1],
})
DIRTY_SNAPSHOT = DIRTY.copy(deep=True)

# ===========================================================================
print("── ① 主判据：三种停牌形态都打标 ──")
try:
    out = run(DIRTY, {}, CTX)
except Exception as e:                          # noqa: BLE001
    print(f"  [!!] suspension.run 调用失败：{type(e).__name__}: {e}")
    sys.exit(1)

if out is None:
    print("  [!!] suspension.run 返回了 None（忘了 return df？）")
    sys.exit(1)

check(out["is_suspended"].tolist() == [0, 1, 1, 1, 0],
      f"is_suspended = {out['is_suspended'].tolist()}（期望 [0,1,1,1,0]）")
check(out["is_suspended"].dtype == "int8", f"is_suspended dtype = {out['is_suspended'].dtype}（期望 int8）")
check(out["is_suspended"].sum() == 3, "停牌 3 行（僵尸/量空/全空各 1）")

print("")
print("── ② 游程：连续第几天，复牌重置 ──")
check(out["suspension_days"].tolist() == [0, 1, 2, 3, 0],
      f"suspension_days = {out['suspension_days'].tolist()}（期望 [0,1,2,3,0]）")
check(out["suspension_days"].dtype == "int16", f"suspension_days dtype = {out['suspension_days'].dtype}（期望 int16）")

print("")
print("── ③ 约束：不删行 / 不改原 df / 只多两列 ──")
check(len(out) == len(DIRTY), f"行数不变：{len(DIRTY)} -> {len(out)}")
check(DIRTY.equals(DIRTY_SNAPSHOT), "原 df 未被修改（copy 生效）")
check(list(out.columns) == list(DIRTY.columns) + ["is_suspended", "suspension_days", "is_resume", "resume_after_days"],
      f"新列追加在末尾：{list(out.columns)[-4:]}")
check(not out["date"].isna().any(), "date 一列没被碰过")

print("")
print("── ④ 记账：attrs['suspension'] 五个计数 ──")
st = out.attrs.get("suspension", {})
check(st.get("suspended_rows") == 3, f"suspended_rows = {st.get('suspended_rows')}（停牌总行数）")
check(st.get("zombie_rows") == 1, f"zombie_rows = {st.get('zombie_rows')}（四价齐全 + volume=0）")
check(st.get("nan_volume_rows") == 1, f"nan_volume_rows = {st.get('nan_volume_rows')}（四价齐全 + volume=NaN）")
check(st.get("empty_rows") == 1, f"empty_rows = {st.get('empty_rows')}（四价全 NaN）")
check(st.get("contradiction_rows") == 0, f"contradiction_rows = {st.get('contradiction_rows')}（停牌却有量）")

print("")
print("── ⑤ 幂等 ──")
check(run(out, {}, CTX).equals(out), "run(run(df)) == run(df)")

print("")
print("── ⑥ 守卫 ──")
expect_raise(lambda: run(pd.DataFrame(), {}, CTX), "空表 → ValueError")
expect_raise(lambda: run(DIRTY.drop(columns=["tradestatus"]), {}, CTX), "缺 tradestatus → ValueError")
expect_raise(lambda: run(DIRTY.drop(columns=["volume"]), {}, CTX), "缺 volume → ValueError（通用必需列）")
_r = expect_ok(lambda: run(DIRTY.drop(columns=["amount"]), {}, CTX), "缺 amount → 正常通过（不是必需列）")
check(_r is not None and list(_r.columns) == list(DIRTY.drop(columns=["amount"]).columns) + ["is_suspended", "suspension_days", "is_resume", "resume_after_days"],
      "缺 amount 时依然只多 4 列")

print("")
print("── ⑦ 边界：游程的极端与矛盾行 ──")
check(run(make_df([0, 0, 0]), {}, CTX)["suspension_days"].tolist() == [1, 2, 3], "全程停牌 -> [1,2,3]")
check(run(make_df([1, 1, 1]), {}, CTX)["suspension_days"].tolist() == [0, 0, 0], "全程正常 -> [0,0,0]")
check(run(make_df([0, 0, 1, 1]), {}, CTX)["suspension_days"].tolist() == [1, 2, 0, 0], "开头就停牌 -> [1,2,0,0]")
check(run(make_df([0, 1, 0]), {}, CTX)["suspension_days"].tolist() == [1, 0, 1], "停-复-停 -> [1,0,1]")
check(run(make_df([1, 0, 1]), {}, CTX)["suspension_days"].tolist() == [0, 1, 0], "单行停牌 -> [0,1,0]")
_c = run(make_df([1, 0], volume=[1000, 500]), {}, CTX)
check(_c.attrs["suspension"]["contradiction_rows"] == 1 and _c["is_suspended"].tolist() == [0, 1],
      "停牌却有成交量 -> contradiction_rows=1，但仍打 is_suspended=1")

print("")
print("── ⑧ 真实数据（缺数据文件时自动跳过）──")
CASES = [
    ("sh.600000", "2015-06-03", "2015-06-12", [0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 2, 3, 4, 5], 5, 0),
    ("sh.600006", "2022-05-23", "2022-05-31", [0, 0, 1, 1, 1, 1, 0], [0, 0, 1, 2, 3, 4, 0], 0, 4),
    ("sz.000022", "2018-12-24", "2018-12-28", None, None, None, None),
]
for code, a, b, exp_susp, exp_days, exp_zombie, exp_nanvol in CASES:
    p = Path(STOCK_DAILY_DIR) / f"{code}.parquet"
    if not p.exists():
        skipped += 1
        print(f"  [--] {code} 数据文件不存在，跳过")
        continue
    d = pd.read_parquet(p)
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["date"] >= a) & (d["date"] <= b)]
    r = run(d, {}, CTX)
    if exp_susp is not None:
        check(r["is_suspended"].tolist() == exp_susp, f"{code} is_suspended = {r['is_suspended'].tolist()}")
        check(r["suspension_days"].tolist() == exp_days, f"{code} suspension_days = {r['suspension_days'].tolist()}")
        check(r.attrs["suspension"]["zombie_rows"] == exp_zombie, f"{code} zombie_rows = {exp_zombie}")
        check(r.attrs["suspension"]["nan_volume_rows"] == exp_nanvol, f"{code} nan_volume_rows = {exp_nanvol}")
    else:
        check(r["is_suspended"].sum() >= 1 and r.attrs["suspension"]["empty_rows"] >= 1,
              f"{code} 全空行形态被识别（empty_rows={r.attrs['suspension']['empty_rows']}）")

# ===========================================================================
# ===========================================================================
# ⑨ 复牌日两列（is_resume / resume_after_days）
#    未实现时自动跳过；一旦实现，本节立刻变成强制断言（不再只是"打印"）
#    契约：is_resume=1 ⇔ 前一行停牌且本行正常
#    不变式：is_resume=1 ⇒ resume_after_days >= 1   （复牌必然对应一段 >=1 天的停牌）
# ===========================================================================
print("")
print("── ⑨ 复牌日两列与不变量 ──")
RESUME_COLS = {"is_resume", "resume_after_days"}
if not RESUME_COLS.issubset(out.columns):
    skipped += 1
    print("  [--] 复牌日两列尚未实现（契约已定义）→ 本节跳过；实现后会自动变成强制断言")
else:
    # ① 首行边界：第一行不可能有"前一行停牌"
    check(out["is_resume"].iloc[0] == 0,
          f"首行 is_resume = {out['is_resume'].iloc[0]}（必须为 0）")
    # ② 不变量：复牌必然对应一段 >=1 天的停牌
    _bad = (out["is_resume"] == 1) & (out["resume_after_days"] < 1)
    check(not _bad.any(),
          f"不变量 is_resume=1 ⇒ resume_after_days>=1（违约 {int(_bad.sum())} 行）")
    # ③ 真实案例：sh.600000 停 7 天后复牌（13 行窗口里只有 1 个复牌日）
    _p = Path(STOCK_DAILY_DIR) / "sh.600000.parquet"
    if _p.exists():
        _d = pd.read_parquet(_p)
        _d["date"] = pd.to_datetime(_d["date"])
        _d = _d[(_d["date"] >= "2015-06-03") & (_d["date"] <= "2015-06-19")]
        _r = run(_d, {}, CTX)
        check(_r["is_resume"].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              f"sh.600000 is_resume = {_r['is_resume'].tolist()}（期望只有 06-17 为 1）")
        check(int(_r["resume_after_days"].max()) == 7,
              f"resume_after_days 最大值 = {int(_r['resume_after_days'].max())}（期望 7）")
    else:
        skipped += 1
        print("  [--] sh.600000 数据不存在，跳过真实案例")

# ⑩ 扩展区 —— 在这里注入你自己的异常与特殊情况
#
#   用法：用 make_df() 造方案，再用 check() 写期望。
#     形态 ->  make_df(tradestatus=[...], volume=[...], price=[...])
#
#   示例（取消注释即可跑）：
#     _e = run(make_df([1, 2, 0]), {}, CTX)
#     check(_e["is_suspended"].tolist() == [0, 0, 1],
#           "记录当前行为：非 0/1 取值被当成'正常'（errors='coerce' 不抛错）")
#
#     _n = run(make_df([1, float("nan"), 1]), {}, CTX)
#     check(_n["is_suspended"].tolist() == [0, 0, 0],
#           "记录当前行为：NaN 状态判为'正常'（待定：是否该 raise）")
# ===========================================================================
print("")
print("── ⑩ 扩展区（你自己的注入）──")
check(True, "扩展区占位（注入后把这行删掉）")

# ===========================================================================
if FULL:
    print("")
    print("── ⑪ 全库对账（约 1 分钟）──")
    tot = susp = 0
    files = sorted(Path(STOCK_DAILY_DIR).glob("*.parquet"))
    for i, f in enumerate(files, 1):
        raw = pd.read_parquet(f)
        r = run(raw, {}, CTX)
        tot += len(r)
        susp += int(r["is_suspended"].sum())
        if i % 1000 == 0:
            print(f"    ... {i}/{len(files)}", flush=True)
    check(tot == 11_641_022, f"总行数 = {tot:,}（期望 11,641,022）")
    check(susp == EXPECTED_LIBRARY_SUSPENDED,
          f"is_suspended.sum() = {susp:,}（期望 {EXPECTED_LIBRARY_SUSPENDED:,}，与体检 C14 对账）")

print("")
print("=" * 46)
print(f"结果: 通过 {passed} / {passed + failed}" + (f"，跳过 {skipped}" if skipped else ""))
if failed == 0:
    print("全部通过 ✅ suspension 停牌环节可用")
else:
    print(f"仍有 {failed} 项失败 ❌ 请按上面提示修复")
    sys.exit(1)
