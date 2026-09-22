"""ops/clean_all.py —— 全库清洗流程观察器（只读，不落盘；跑完给出全库对账）

用途：亲眼看一遍清洗管道在跑 —— 每 N 只打一行进度，最后汇总各环节的 attrs 统计，
      并在"全库 + 默认区间"时直接和已知实测值对账（✅ / ❌）。

跑法（项目根目录）：
    python -m ops.clean_all                    # 全库 5,471 只，约 150 秒
    python -m ops.clean_all --limit 200        # 先跑 200 只看格式（约 4 秒）
    python -m ops.clean_all --every 1000       # 进度行稀疏一点
    python -m ops.clean_all --quiet            # 只打最后的汇总

不做：不写缓存、不改任何文件。要生成面板缓存请用 backtest/panel.py。
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data import loader
from data.cleaner import pipeline

START, END = "2015-01-01", "2026-12-31"

# 各环节 attrs 里要汇总的键：(环节, 键)
FIELDS = [
    ("suspension", "suspended_rows"), ("suspension", "zombie_rows"),
    ("suspension", "nan_volume_rows"), ("suspension", "empty_rows"),
    ("suspension", "contradiction_rows"),
    ("price", "price_bad_rows"), ("price", "amt_vol_bad_rows"),
    ("price", "ohlc_violation_rows"),
    ("adjust", "hfq_nan_rows"), ("adjust", "factor_nonpositive_rows"),
    ("derived", "ret_nan_rows"), ("derived", "ret_inf_rows"),
]

# 全库 + 默认区间时的实测对账目标（None = 不校验）
EXPECT = {
    "rows": 11_641_022,
    "suspended_rows": 282_893,
    "price_bad_rows": 3,
    "amt_vol_bad_rows": 60,
    "hfq_nan_rows": 3,
    "ret_nan_rows": 5_471,
    "ret_inf_rows": 0,
    "warnings": 0,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 只（默认全库）")
    ap.add_argument("--every", type=int, default=500, help="每 N 只打一行进度")
    ap.add_argument("--quiet", action="store_true", help="只打最后的汇总")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    a = ap.parse_args()

    codes = loader.list_codes()
    if a.limit:
        codes = codes[: a.limit]
    full = a.limit is None and a.start == START and a.end == END

    print(f"标的 {len(codes)} 只 | 区间 {a.start} ~ {a.end} | 环节 " +
          " → ".join(n for n, _ in pipeline.PIPELINE))
    if not a.quiet:
        print("-" * 100)

    agg: dict[str, int] = collections.defaultdict(int)
    cols: dict[str, set] = collections.defaultdict(set)
    warn = 0
    t0 = time.time()
    for i, (code, df) in enumerate(loader.iter_frames(codes, a.start, a.end), 1):
        out, reps = pipeline.run_one(
            df, {"universe": "stock", "asset_type": "stock", "code": code})
        for r in reps:
            cols[r["name"]].update(r["columns_added"])
            warn += len(r["warnings"])
        agg["rows"] += len(out)
        for stage, key in FIELDS:
            agg[key] += int(out.attrs.get(stage, {}).get(key, 0))
        agg["stocks"] = i
        if not a.quiet and (i <= 2 or i % a.every == 0 or i == len(codes)):
            dt = time.time() - t0
            added = ",".join(c for n in ("suspension", "price", "adjust", "derived")
                             for c in sorted(cols[n]))
            print(f"[{i:>5}/{len(codes)}] {code}  行 {len(out):>5}  "
                  f"新增列 {added}  用时 {dt:6.1f}s  速度 {i / max(dt, 1e-9):5.1f} 只/秒")

    print("-" * 100)
    print(f"合计：{agg['stocks']:,} 只 | {agg['rows']:,} 行 | 用时 {time.time() - t0:.0f}s")
    print("\n各环节新增列（必须与白名单一致）：")
    for n in ("align", "suspension", "price", "adjust", "derived"):
        print(f"  {n:<11}{sorted(cols[n])}")
    print(f"\n未登记列警告数：{warn}")

    print("\nattrs 汇总：" + ("" if not full else "（右侧为全库实测目标）"))
    bad = []
    for k in ([f[1] for f in dict.fromkeys(FIELDS)] + ["rows", "warnings"]):
        v = warn if k == "warnings" else agg.get(k, 0)
        if full and k in EXPECT:
            ok = v == EXPECT[k]
            bad += [] if ok else [k]
            print(f"  {k:<26}{v:>13,}   {'✅' if ok else '❌'} 目标 {EXPECT[k]:,}")
        else:
            print(f"  {k:<26}{v:>13,}")

    if full:
        print("\n" + ("✅ 全库对账全部命中" if not bad else f"❌ 不匹配：{bad}"))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
