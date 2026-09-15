"""
ETF 数据测试工具
===============
验证本地 Parquet 数据是否完整、格式是否正确。
用法:
  python "legacy_data_check.py"                  # 列出本地 ETF，挑第一只全面测试
  python "legacy_data_check.py" sh510050          # 测试指定 ETF 日线
  python "legacy_data_check.py" sh510050 60min    # 测试指定 ETF 60分钟线
  python "legacy_data_check.py" --list            # 仅列出本地已有的 ETF
  python "legacy_data_check.py" --all             # 扫描全部本地 ETF，报告数据质量
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ============================================================================
# 路径常量（自行计算，不依赖 config 模块，避免 linter 重排 import 导致
# config 在 sys.path 修正之前就被导入的问题）
# ============================================================================

_PROJECT_ROOT = Path(__file__).resolve(
).parent.parent  # tests → quant 1.0
_DATA_STORAGE = _PROJECT_ROOT / "data" / "storage"
ETF_DAILY_DIR = _DATA_STORAGE / "etf_daily"
ETF_60MIN_DIR = _DATA_STORAGE / "etf_60min"

# ============================================================================
# 配置
# ============================================================================

REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]
FREQ_MAP = {"d": ("日线", ETF_DAILY_DIR), "60min": ("60分钟线", ETF_60MIN_DIR)}


# ============================================================================
# 核心函数
# ============================================================================

def list_local_etfs(freq: str = "d") -> list[str]:
    """返回本地已有的 ETF 代码列表"""
    _, d = FREQ_MAP.get(freq, (None, None))
    if d is None or not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.parquet"))


def load_one(code: str, freq: str = "d") -> pd.DataFrame:
    """读取单只 ETF 的本地 Parquet"""
    _, d = FREQ_MAP.get(freq, (None, None))
    if d is None:
        raise ValueError(f"不支持的频率: {freq}，可选 d / 60min")
    fp = d / f"{code}.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"本地无 {code} 的 {freq} 数据: {fp}")
    df = pd.read_parquet(fp)
    if "date" in df.columns and not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def test_one(df: pd.DataFrame, code: str, freq: str) -> dict:
    """对单只 ETF 的数据运行一系列检查，返回结果字典"""
    label, _ = FREQ_MAP.get(freq, (freq, None))
    result = {"code": code, "freq": freq,
              "label": label, "pass": True, "issues": []}

    # 检查1: 非空
    if df.empty:
        result["pass"] = False
        result["issues"].append("[FAIL] 数据为空")
        return result
    result["rows"] = len(df)

    # 检查2: 必有列
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        result["pass"] = False
        result["issues"].append(f"[FAIL] 缺少必需列: {missing_cols}")

    # 检查3: 日期格式
    if "date" in df.columns:
        null_dates = df["date"].isna().sum()
        if null_dates > 0:
            result["pass"] = False
            result["issues"].append(f"[FAIL] date 列有 {null_dates} 个空值")
        result["date_min"] = df["date"].min()
        result["date_max"] = df["date"].max()

    # 检查4: OHLCV 全空行
    ohlcv_cols = [c for c in REQUIRED_COLS[1:] if c in df.columns]
    if ohlcv_cols:
        all_null = df[ohlcv_cols].isna().all(axis=1).sum()
        if all_null > 0:
            result["issues"].append(f"[WARN] {all_null} 行 OHLCV 全为空")

    # 检查5: 重复日期
    if "date" in df.columns:
        dup_dates = df["date"].duplicated().sum()
        if dup_dates > 0:
            result["issues"].append(f"[WARN] {dup_dates} 个重复日期")

    # 检查6: 日期连续性（日线）
    if freq == "d" and "date" in df.columns and len(df) > 5:
        expected_days = (df["date"].max() - df["date"].min()).days
        actual_days = len(df)
        if expected_days > 0:
            coverage = actual_days / expected_days
            if coverage < 0.6:
                result["issues"].append(
                    f"[WARN] 日期覆盖率仅 {coverage:.0%} ({actual_days}/{expected_days} 天)")

    # 检查7: 异常值（价格为0或负数）
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            bad = (df[col] <= 0).sum()
            if bad > 0:
                result["pass"] = False
                result["issues"].append(f"[FAIL] {col} 有 {bad} 个 ≤0 的值")

    if "volume" in df.columns:
        neg_vol = (df["volume"] < 0).sum()
        if neg_vol > 0:
            result["issues"].append(f"[WARN] volume 有 {neg_vol} 个负值")

    # 检查8: high < low
    if "high" in df.columns and "low" in df.columns:
        bad_hl = (df["high"] < df["low"]).sum()
        if bad_hl > 0:
            result["pass"] = False
            result["issues"].append(f"[FAIL] {bad_hl} 行 high < low")

    return result


def preview(df: pd.DataFrame, code: str, freq: str) -> None:
    """打印数据预览"""
    label, _ = FREQ_MAP.get(freq, (freq, None))
    if df.empty:
        print(f"{code} ({label}): 空数据")
        return

    print(f"\n{'=' * 60}")
    print(f"ETF: {code}  频率: {label}  行数: {len(df)}")
    print(f"日期: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"列:   {df.columns.tolist()}")
    print(f"{'=' * 60}")
    print("\n── 头部 10 行 ──")
    print(df.head(10).to_string(index=False))
    print("\n── 尾部 5 行 ──")
    print(df.tail(5).to_string(index=False))

    num_cols = [c for c in ["open", "high", "low",
                            "close", "volume", "amount"] if c in df.columns]
    if num_cols:
        print("\n── 描述统计 ──")
        print(df[num_cols].describe().to_string())


def print_test_result(r: dict) -> None:
    """打印单项测试结果"""
    status = "[OK] 通过" if r["pass"] else "[FAIL] 失败"
    extra = f"  |  {r.get('rows', '?')} 行  |  {r.get('date_min', '?')} ~ {r.get('date_max', '?')}" if "rows" in r else ""
    print(f"  {r['code']} ({r['label']})  {status}{extra}")
    for issue in r["issues"]:
        print(f"    {issue}")


# ============================================================================
# 主入口
# ============================================================================

def main() -> None:
    args = sys.argv[1:]

    # ── --list ──
    if "--list" in args:
        for freq_key, (label, d) in FREQ_MAP.items():
            codes = list_local_etfs(freq_key)
            print(f"\n本地 {label} ({len(codes)} 只):")
            if codes:
                print(", ".join(codes[:60]))
                if len(codes) > 60:
                    print(f"... 还有 {len(codes) - 60} 只")
        return

    # ── --all：扫描全部 ──
    if "--all" in args:
        all_ok, all_fail = 0, 0
        for freq_key, (label, d) in FREQ_MAP.items():
            codes = list_local_etfs(freq_key)
            if not codes:
                print(f"\n{label}: 无本地数据")
                continue
            print(f"\n{'=' * 60}")
            print(f"扫描 {label}: {len(codes)} 只")
            print(f"{'=' * 60}")
            for code in codes:
                try:
                    df = load_one(code, freq_key)
                    r = test_one(df, code, freq_key)
                    if r["pass"]:
                        all_ok += 1
                    else:
                        all_fail += 1
                    if r["issues"]:
                        print_test_result(r)
                except Exception as e:
                    all_fail += 1
                    print(f"  {code} ({label}) [FAIL] 读取失败: {e}")
        print(f"\n{'=' * 60}")
        print(f"总计: {all_ok} 通过, {all_fail} 失败")
        return

    # ── 单只测试 ──
    freq = "d"
    code = None

    for a in args:
        if a in ("d", "60min"):
            freq = a
        elif not a.startswith("--"):
            code = a

    if code is None:
        codes = list_local_etfs(freq)
        if not codes:
            print(
                f"本地还没有 {FREQ_MAP[freq][0]} 数据，请先运行 baostock_etf_fetcher.py 下载。")
            return
        code = codes[0]
        print(f"未指定代码，自动选择: {code}")
        print(
            f"可用 ({len(codes)} 只): {', '.join(codes[:20])}{'...' if len(codes) > 20 else ''}")

    try:
        df = load_one(code, freq)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    r = test_one(df, code, freq)
    print_test_result(r)
    print()
    preview(df, code, freq)


if __name__ == "__main__":
    main()


# 测试用命令
  # python "F:\quant 1.0\tests\legacy_data_check.py" --list

  # python "F:\quant 1.0\tests\legacy_data_check.py" sh510050

  # python "F:\quant 1.0\tests\legacy_data_check.py" --all
