# -*- coding: utf-8 -*-
"""
align.py 验收测试（脏数据矩阵版）

覆盖 align 的四步 + 三条约束 + 两类守卫：

  ① 日期规范化     字符串日期 → datetime64；解析失败的行删掉（唯一允许删行的场景）
  ② 同日去重       keep="first"：保留**原顺序里先出现**的那行
  ③ 升序排序       date 单调递增、无重复
  ④ 数值列类型修正 字符串数字 → float64；坏值 → NaN；**绝不 fillna**

  约束：不改原 df（copy）｜不加新列｜幂等（跑两遍结果一样）
  守卫：空表 → ValueError；缺必需列 → ValueError

⚠️ 数据是故意造脏的：字符串日期、字符串数字、坏值 "abc"、缺失 None、乱序、同日重复。
   —— 干净数据下"类型转换"那几行永远不会被真正执行（对应 PITFALLS #8/#10）。

用法（项目根目录）:
  python "tests/test_align.py"
期望末尾: 全部通过 (X/Y)
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from data.cleaner.align import run

CTX = {"universe": "stock", "asset_type": "stock", "code": "TEST"}

# ---------------------------------------------------------------------------
# 测试数据 ①：四步全脏 —— 字符串日期 + 乱序 + 同日重复(值不同) + 坏日期 + None
#              数字全部写成字符串（逼 to_numeric 那几行执行）
#              故意不放 amount 列（验证 "if col in out.columns" 的容错）
# ---------------------------------------------------------------------------
DIRTY = pd.DataFrame({
    "date":   ["2026-01-02", "2026-01-01", "2026-01-02", "bad-date", None, "2026-01-03"],
    "open":   ["2.0", "1.0", "9.9", "5.0", "5.0", "3.0"],
    "high":   ["2.2", "1.2", "9.9", "5.0", "5.0", "3.2"],
    "low":    ["1.9", "0.9", "9.9", "5.0", "5.0", "2.8"],
    "close":  ["2.1", "1.05", "9.9", "5.0", "5.0", "3.1"],
    "volume": ["200", "100", "999", "0", "0", "150"],
})
DIRTY_SNAPSHOT = DIRTY.copy(deep=True)

# ---------------------------------------------------------------------------
# 测试数据 ②：数值坏值/缺失 —— "abc" 与 None 必须变成 NaN，且**不允许被填成 0**
# ---------------------------------------------------------------------------
NUM_DIRTY = pd.DataFrame({
    "date":   ["2026-01-01", "2026-01-02", "2026-01-03"],
    "open":   ["1.0", "abc", "3.0"],
    "high":   ["1.1", "2.2", "3.2"],
    "low":    ["0.9", "1.8", "2.8"],
    "close":  ["1.05", None, "3.05"],
    "volume": ["100", "200", "300"],
})

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


def expect_raise(fn, label, err=ValueError):
    global passed, failed
    try:
        fn()
    except err:
        passed += 1
        print(f"  ✅ {label}")
    except Exception as e:                      # noqa: BLE001
        failed += 1
        print(f"  ❌ {label}: 抛了 {type(e).__name__} 而非 {err.__name__}: {e}")
    else:
        failed += 1
        print(f"  ❌ {label}: 期望抛 {err.__name__}，但没有抛")


# ===========================================================================
print("── ① 日期规范化 + 删坏日期行 ──")
try:
    out = run(DIRTY, {}, CTX)
except Exception as e:                          # noqa: BLE001
    print(f"  ❌ align.run 调用失败：{type(e).__name__}: {e}")
    print("     （若是 TypeError: 'NoneType' object is not subscriptable → align 还是空壳，先写实现）")
    sys.exit(1)

if out is None:
    print("  ❌ align.run 返回了 None（忘了 return df？）")
    sys.exit(1)

check(out["date"].dtype == "datetime64[ns]", f"date dtype = {out['date'].dtype}")
check(len(out) == 3, f"坏日期行被删：6 行 → {len(out)} 行（bad-date / None 各删 1 行）")
check(list(out["date"].dt.strftime("%Y-%m-%d")) == ["2026-01-01", "2026-01-02", "2026-01-03"],
      f"剩下的日期: {list(out['date'].dt.strftime('%Y-%m-%d'))}")

print("\n── ② 同日去重 keep='first' ──")
check(float(out.loc[out["date"] == "2026-01-02", "open"].iloc[0]) == 2.0,
      "01-02 保留的是**原顺序先出现**的那行（open=2.0，而不是 9.9）")
check(not bool(out["date"].duplicated().any()), "date 无重复")
check(len(out) == out["date"].nunique(), "行数 == 唯一日期数")

print("\n── ③ 升序排序 + 索引重置 ──")
check(bool(out["date"].is_monotonic_increasing), "date 升序")
check(list(out.index) == list(range(len(out))), f"索引连续: {list(out.index)}")

print("\n── ④ 数值列类型修正（脏数据里全是字符串）──")
# 契约是"数值列可参与计算"，不是"必须是 float64"——整数串 "200" 转成 int64 同样满足契约
for col in ["open", "high", "low", "close", "volume"]:
    check(pd.api.types.is_numeric_dtype(out[col]) and out[col].dtype != object,
          f"{col} 已是数值类型: {out[col].dtype}")

print("\n── ⑤ 约束：不改原 df / 不加新列 ──")
check(DIRTY.equals(DIRTY_SNAPSHOT), "原 df 未被修改（copy 生效）")
check(set(out.columns) == set(DIRTY.columns), f"列集合不变（没偷偷加列）: {sorted(out.columns)}")

print("\n── ⑥ 坏值/缺失 → NaN，且不 fillna ──")
num_out = run(NUM_DIRTY, {}, CTX)
check(num_out["open"].dtype == "float64", f"open dtype = {num_out['open'].dtype}")
check(bool(pd.isna(num_out.loc[1, "open"])), "坏值 'abc' → NaN（errors='coerce'）")
check(bool(pd.isna(num_out.loc[1, "close"])), "缺失 None → NaN")
check(len(num_out) == 3, f"数值坏值不删行：仍 {len(num_out)} 行")
check(not bool((num_out[["open", "close"]].fillna(-999) == 0).any().any()),
      "没有被 fillna(0)：NaN 仍是 NaN，没变成 0")

print("\n── ⑦ 幂等：同一输入跑两遍结果一致 ──")
out1 = run(DIRTY, {}, CTX)
out2 = run(out1, {}, CTX)
check(out1.equals(out2), "run(run(df)) == run(df)")

print("\n── ⑧ 守卫：空表 / 缺必需列 ──")
expect_raise(lambda: run(pd.DataFrame(), {}, CTX), "空表 → ValueError")
expect_raise(lambda: run(DIRTY.drop(columns=["volume"]), {}, CTX), "缺 volume 列 → ValueError")

print("\n" + "=" * 46)
print(f"结果: 通过 {passed} / {passed + failed}")
if failed == 0:
    print("全部通过 ✅ align 整备环节可用")
else:
    print(f"仍有 {failed} 项失败 ❌ 请按上面提示修复")
    sys.exit(1)
