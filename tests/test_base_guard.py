# -*- coding: utf-8 -*-
"""
base.py 守卫 / 翻译官的最小冒烟测试
（内容来自教学文档 `base.py教学参考.md` 第 4 节，用 try/except 包起来，
  保证 4 条能一次跑完并打印结果；正式的 15 项验收在 test_base.py 里）

用法（在项目根目录任意终端）:
  python tests/test_base_guard.py
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. 路径准备：必须在 import data.cleaner 之前完成（import 是运行时语句）
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from data.cleaner import base

passed = 0
failed = 0


def expect_raise(fn, label, err_type=ValueError):
    """期望 fn() 抛指定异常"""
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
    """期望 fn() 安静通过"""
    global passed, failed
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        failed += 1
        print(f"  ❌ {label}: 不该抛却抛了 {type(e).__name__}: {e}")
    else:
        passed += 1
        print(f"  ✅ {label}")


def check_eq(actual, expected, label):
    global passed, failed
    if actual == expected:
        passed += 1
        print(f"  ✅ {label}: {actual}")
    else:
        failed += 1
        print(f"  ❌ {label}: 期望 {expected}, 实际 {actual}")


print("── ① 空表: 应该报错 ──")
expect_raise(lambda: base.check_not_empty(pd.DataFrame()),
             "check_not_empty(空df) 抛 ValueError")

print("\n── ② 缺列: 应该报错并说出缺谁 ──")
expect_raise(lambda: base.check_required_columns(pd.DataFrame({"date": ["2026-01-01"]})),
             "缺 open/high/low/close/volume 抛 ValueError")

print("\n── ③ 正常表: 应该安静通过 ──")
df_ok = pd.DataFrame({
    "date": ["2026-01-01"], "open": [1.0], "high": [1.1],
    "low": [0.9], "close": [1.05], "volume": [100],
})
expect_ok(lambda: base.check_not_empty(df_ok), "check_not_empty(正常表) 通过")
expect_ok(lambda: base.check_required_columns(df_ok), "check_required_columns(正常表) 通过")

print("\n── ④ 翻译官 ──")
check_eq(base.as_int("1.0", 0), 1, "as_int('1.0', 0)")
check_eq(base.as_int("abc", 5), 5, "as_int('abc', 5) 翻不动用默认")

print("\n" + "=" * 40)
print(f"结果: 通过 {passed} / {passed + failed}")
if failed == 0:
    print("冒烟测试全部通过 ✅")
else:
    print(f"仍有 {failed} 项失败 ❌")
    sys.exit(1)
