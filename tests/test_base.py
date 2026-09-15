# -*- coding: utf-8 -*-
"""
base.py 验收测试：验证两件事
  1) 可以跨文件调用（从别的文件 import 并调用，不炸）
  2) 内部逻辑可用（守卫/翻译/白名单检查行为正确）

用法（在项目根目录任意终端）:
  python tests/test_base.py
期望输出末尾: 全部通过 (X/Y)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. 路径准备：保证能从项目根 import data.cleaner.base
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

# ---------------------------------------------------------------------------
# 1. 跨文件调用测试：本测试文件本身就是"另一个文件"
# ---------------------------------------------------------------------------
try:
    from data.cleaner import base
except Exception as e:  # noqa: BLE001
    print(f"[跨文件调用] ❌ 无法 import data.cleaner.base: {type(e).__name__}: {e}")
    print("  请确认 base.py 已创建在 data/cleaner/base.py，且无语法/名字错误")
    sys.exit(1)

# 需要的函数清单（教学规格里的核心件）
NEEDED = ["REQUIRED_COLUMNS", "COLUMN_WHITELIST",
          "check_not_empty", "check_required_columns",
          "check_new_columns", "as_int", "as_float"]
missing_api = [n for n in NEEDED if not hasattr(base, n)]
if missing_api:
    print(f"[跨文件调用] ❌ base.py 还缺这些名字(函数/常量): {missing_api}")
    sys.exit(1)
print("[跨文件调用] ✅ import 成功，所需 API 齐全")


# 模拟"别的文件"里写的一个调用方函数（证明跨文件调用可用）
def caller_guard(df, required=None):
    """这是另一个文件里的函数——它 import 了 base 并调用"""
    base.check_not_empty(df, context="caller")
    base.check_required_columns(df, required=required)


# ---------------------------------------------------------------------------
# 2. 内部逻辑测试
# ---------------------------------------------------------------------------
passed = 0
failed = 0


def expect_raise(fn, label, err_type=ValueError):
    """期望 fn() 抛异常"""
    global passed, failed
    try:
        fn()
    except err_type:
        passed += 1
        print(f"  ✅ {label}")
    except Exception as e:  # noqa: BLE001
        failed += 1
        print(f"  ❌ {label}: 抛了 {type(e).__name__} 而非 {err_type.__name__}: {e}")
    else:
        failed += 1
        print(f"  ❌ {label}: 期望抛 {err_type.__name__}，但没有抛")


def expect_ok(fn, label):
    """期望 fn() 不抛异常"""
    global passed, failed
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        failed += 1
        print(f"  ❌ {label}: 不该抛却抛了 {type(e).__name__}: {e}")
    else:
        passed += 1
        print(f"  ✅ {label}")


# 正常表（满足 REQUIRED_COLUMNS）
df_ok = pd.DataFrame({
    "date": ["2026-01-01", "2026-01-02"],
    "open": [1.0, 1.1], "high": [1.1, 1.2],
    "low": [0.9, 1.0], "close": [1.05, 1.15], "volume": [100, 200],
})

print("\n── 守卫: 空表 ──")
expect_raise(lambda: base.check_not_empty(None), "check_not_empty(None) 抛错")
expect_raise(lambda: base.check_not_empty(pd.DataFrame()), "check_not_empty(空df) 抛错")
expect_raise(lambda: caller_guard(pd.DataFrame()), "跨文件调用: 空表抛错")
expect_ok(lambda: base.check_not_empty(df_ok), "check_not_empty(正常表) 通过")

print("\n── 守卫: 缺列 ──")
df_missing = df_ok.drop(columns=["volume"])
expect_raise(lambda: base.check_required_columns(df_missing), "缺 volume 抛错")
expect_raise(lambda: base.check_required_columns(df_ok, ["back_adj_factor"]),
             "自定义 required 缺 back_adj_factor 抛错")
expect_ok(lambda: base.check_required_columns(df_ok), "列齐全 通过")
expect_ok(lambda: caller_guard(df_ok), "跨文件调用: 正常表通过")

print("\n── 参数翻译官 ──")


# 用普通 assert 而非上面的 lambda 技巧，更直观
def check_eq(actual, expected, label):
    global passed, failed
    if actual == expected:
        passed += 1
        print(f"  ✅ {label}: {actual}")
    else:
        failed += 1
        print(f"  ❌ {label}: 期望 {expected}, 实际 {actual}")


check_eq(base.as_int("1.0", 0), 1, "as_int('1.0', 0) == 1")
check_eq(base.as_int("abc", 5), 5, "as_int('abc', 5) == 5 (翻不动用默认)")
check_eq(base.as_int(None, 7), 7, "as_int(None, 7) == 7")
check_eq(base.as_float("2.5", 0.0), 2.5, "as_float('2.5') == 2.5")

print("\n── 列白名单检查 ──")
before = ["date", "open", "high", "low", "close", "volume"]
check_eq(base.check_new_columns(before, before), [], "没加新列 → []")
# 白名单里登记过的列 → [] (假设 is_suspended 已在白名单)
check_eq(base.check_new_columns(before, before + ["is_suspended"]), [],
         "加了白名单列 is_suspended → []")
# 白名单外的列 → 提醒
check_eq(base.check_new_columns(before, before + ["is_suspend"]), ["is_suspend"],
         "加了未登记列 is_suspend → 提醒 ['is_suspend']")

# ---------------------------------------------------------------------------
# 3. 汇总
# ---------------------------------------------------------------------------
print("\n" + "=" * 46)
print(f"结果: 通过 {passed} / {passed + failed}")
if failed == 0:
    print("全部通过 ✅ base.py 可跨文件调用且逻辑可用")
else:
    print(f"仍有 {failed} 项失败 ❌ 请按上面提示修复")
    sys.exit(1)
