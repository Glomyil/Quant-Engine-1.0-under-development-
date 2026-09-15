# -*- coding: utf-8 -*-
"""
L4 实测：akshare 东财 fund_etf_hist_em 的复权口径验证（只读，不落盘）
====================================================================
背景：
  用户已拍板「东财源 + 后复权 hfq 主数据 + raw 归档」。
  但 dsh 沙箱连不通东财域名（push2his.eastmoney.com 被拒），
  需要在本机真实终端运行此脚本，验证 4 件事后把输出贴回对话：

  ① 接口字段长什么样（日期/开盘/收盘/最高/最低/成交量/成交额 …）
  ② 不复权 / qfq / hfq 三档是否都取得到
  ③ hfq 口径自洽性：后复权收盘价应 ≥ 不复权收盘价（ratio ≥ 1），
     否则该源对 ETF 的复权不可信（腾讯源就是反例：ratio<1）
  ④ 已知除息日（510300 → 2026-01-19）在 hfq 下不应再出现跳空，
     即当日 hfq 收益率 ≈ 0% 左右而不是 -2.49%

用法（在本机终端，非 dsh 沙箱）：
  python "tests/l4_em_hfq_probe.py"

无网络时本脚本会打印错误退出，不影响任何本地文件。
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pandas as pd

socket.setdefaulttimeout(30)

# 确保项目根目录在 sys.path（其实本脚本不依赖项目模块，仅为风格统一）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 测试对象：宽基 + 科创板，横跨不同涨跌幅池
SAMPLE: dict[str, str] = {
    "510300": "沪深300ETF",
    "510050": "上证50ETF",
    "159919": "沪深300ETF(嘉实)",
    "588000": "科创50ETF",
}

# 覆盖已知除息日 2026-01-19(510300) 的观察窗口
WIN_START = "20251215"
WIN_END = "20260210"

KNOWN_EX_DIV = {"510300": "2026-01-19", "510050": "2025-12-17", "159919": "2025-12-08"}

RENAME = {"日期": "date", "开盘": "open", "收盘": "close",
          "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"}


def fetch_em(symbol: str, adjust: str) -> pd.DataFrame:
    """拉东财日线；adjust: '' 不复权 / 'qfq' 前复权 / 'hfq' 后复权"""
    import akshare as ak
    df = ak.fund_etf_hist_em(
        symbol=symbol, period="daily",
        start_date=WIN_START, end_date=WIN_END, adjust=adjust,
    )
    if df is None or df.empty:
        raise ValueError(f"{symbol} adjust={adjust!r} 返回空")
    return df.rename(columns=RENAME)


def main() -> int:
    print("=" * 72)
    print("L4 东财复权实测（只读）  观察窗口:", f"{WIN_START}~{WIN_END}")
    print("=" * 72)

    try:
        import akshare  # noqa: F401  提前 import，网络不通时立刻报错
    except Exception as e:
        print(f"[FATAL] 无法 import akshare: {e}")
        return 1

    for symbol, name in SAMPLE.items():
        print(f"\n########## {symbol} {name} ##########")
        frames: dict[str, pd.DataFrame] = {}
        ok = True
        for adj, label in [("", "raw"), ("qfq", "qfq"), ("hfq", "hfq")]:
            try:
                df = fetch_em(symbol, adj)
                frames[adj] = df
                print(f"[OK] {label:>3}: {len(df)} 行  列={df.columns.tolist()}")
            except Exception as e:
                ok = False
                print(f"[FAIL] {label:>3}: {type(e).__name__}: {e}")
        if not ok or len(frames) < 2:
            print("  → 跳过该 ETF 的比对（接口失败）")
            continue

        # ① 字段与类型预览（取 raw 档）
        raw = frames[""].copy()
        raw["date"] = raw["date"].astype(str)
        print("\n[字段示例 raw 尾部 3 行]")
        print(raw[["date", "open", "close", "high", "low", "volume"]].tail(3).to_string(index=False))

        # ② 口径自洽：hfq 应 ≥ raw（后复权含分红再投）
        last_day = raw["date"].iloc[-1]
        r_raw = float(raw.loc[raw["date"] == last_day, "close"].iloc[0])
        r_hfq = float(frames["hfq"].loc[frames["hfq"]["date"].astype(str) == last_day, "close"].iloc[0]) \
            if last_day in frames["hfq"]["date"].astype(str).values else float("nan")
        ratio = r_hfq / r_raw if pd.notna(r_hfq) and r_raw else float("nan")
        verdict = "OK(≥1)" if ratio >= 1.0 else "可疑(<1, 源不可信)"
        print(f"[口径自洽] 末日 {last_day}: raw={r_raw}  hfq={r_hfq}  hfq/raw={ratio:.4f} → {verdict}")

        # ③ 已知除息日验证（只有配了已知日期的样本才查）
        ex = KNOWN_EX_DIV.get(symbol)
        if ex and ex in frames[""].astype(str).to_dict("list").get("date", []):
            row = {}
            for adj, label in [("", "raw"), ("qfq", "qfq"), ("hfq", "hfq")]:
                d = frames[adj].copy()
                d["date"] = d["date"].astype(str)
                d = d.sort_values("date").reset_index(drop=True)
                d["ret"] = d["close"].pct_change() * 100
                hit = d.loc[d["date"] == ex]
                if len(hit):
                    i = hit.index[0]
                    row[label] = (float(hit["close"].iloc[0]), float(d["ret"].iloc[i]))
            if row:
                print(f"[除息日 {ex}] raw(close,ret%)={row.get('raw')}  "
                      f"qfq={row.get('qfq')}  hfq={row.get('hfq')}")
                r = row.get("hfq", (None, None))[1]
                print(f"  → hfq 当日收益率 = {r:.2f}%"
                      + ("  （≈0，除息坑已消除 ✓）" if r is not None and abs(r) < 1 else "  （仍异常？）"))

    print("\n" + "=" * 72)
    print("完成。请把以上输出完整贴回对话。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
