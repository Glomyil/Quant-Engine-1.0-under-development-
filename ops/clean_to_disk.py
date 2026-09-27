"""ops/clean_to_disk.py —— 把 5 个清洗环节的产物落成"净数据"（架构图 ③ 写回 parquet）

跑法（项目根目录）：
    python -m ops.clean_to_disk                   # 全库 5,471 只 → data/storage/cleaner_daily/
    python -m ops.clean_to_disk --limit 200       # 先跑 200 只看格式（约 6 秒）
    python -m ops.clean_to_disk --codes sh.600000,sz.000001
    python -m ops.clean_to_disk --skip-existing   # 目标文件比源文件新 → 跳过
    python -m ops.clean_to_disk --out data/storage/_smoke --meta data/storage/_smoke_meta.json

产物：<out>/<code>.parquet（**原子写**：先写 .parquet.tmp → os.replace 改名）
      meta json（总账：行数/体积/attrs 汇总/失败清单/写入清单/源指纹）

全库预期：5,471 只 / 11,641,022 行 / **20 列** / ≈660–700 MB（上次全库含写盘实测 338 s：清洗 ~150 s + 写盘 ~190 s）

铁律：原始数据只读；单只失败记账 + 继续；**有任何失败 → 退出码 1**。
口径：attrs 汇总字段与 ops/clean_all.py **共用同一张表**（否则两个脚本永远对不上账）。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import constants as C
from data import loader
from data.cleaner import pipeline
from ops.clean_all import END, FIELDS as AGG_FIELDS, START   # 观测与落盘共用一张口径表

# Windows 控制台默认 GBK：✅/❌/→ 这些字符会直接让脚本崩在最后一行 print
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STAGES = ("align", "suspension", "price", "adjust", "derived")


def store_one(out, dest: Path, code: str) -> int:
    """原子写：先写 .tmp，成功后再改名（同盘 rename 原子，被打断只留 .tmp）。"""
    tmp = dest / f"{code}.parquet.tmp"
    final = dest / f"{code}.parquet"
    out.to_parquet(tmp, index=False, compression=C.PARQUET_COMPRESSION)
    os.replace(tmp, final)
    return final.stat().st_size


def source_fingerprint(files) -> dict:
    return {"files": len(files),
            "max_mtime": datetime.fromtimestamp(max(p.stat().st_mtime for p in files))
                                 .strftime("%Y-%m-%d %H:%M:%S"),
            "total_mb": round(sum(p.stat().st_size for p in files) / 2**20, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 只（默认全库）")
    ap.add_argument("--codes", default=None, help="逗号分隔的 code 清单（与 --limit 二选一）")
    ap.add_argument("--src", default=None, help="源目录（默认 config 的 STOCK_DAILY_DIR；注入测试时指假源）")
    ap.add_argument("--out", default=None, help="目标目录（默认 config 的 CLEANER_DAILY_DIR）")
    ap.add_argument("--meta", default=None, help="总账路径（默认 <目标目录的上级>/cleaner_meta.json）")
    ap.add_argument("--skip-existing", action="store_true", help="目标文件比源文件新 → 跳过")
    ap.add_argument("--every", type=int, default=1000, help="每 N 只打一行进度")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    a = ap.parse_args()

    dest = Path(a.out) if a.out else C.CLEANER_DAILY_DIR
    meta_p = Path(a.meta) if a.meta else dest.parent / "cleaner_meta.json"
    dest.mkdir(parents=True, exist_ok=True)

    src_dir = Path(a.src) if a.src else C.STOCK_DAILY_DIR
    all_codes = loader.list_codes(src_dir)
    if a.codes:
        want = {c.strip() for c in a.codes.split(",") if c.strip()}
        codes = [c for c in all_codes if c in want]
    else:
        codes = all_codes[: a.limit] if a.limit else all_codes

    print(f"源 {src_dir}   输出 {dest}   总账 {meta_p}")
    print(f"标的 {len(codes)}/{len(all_codes)} 只 | 区间 {a.start} ~ {a.end} | 环节 "
          + " → ".join(n for n, _ in pipeline.PIPELINE) + "\n" + "-" * 100)

    agg: dict[str, int] = collections.defaultdict(int)
    cols: dict[str, set] = collections.defaultdict(set)
    failures, written, skipped, empty = [], [], 0, 0
    t0 = time.time()
    for i, code in enumerate(codes, 1):   # 逐只处理：**读文件也在 try 里** → 单只坏文件不中断全库
        src = src_dir / f"{code}.parquet"
        final = dest / f"{code}.parquet"
        if a.skip_existing and final.exists() and final.stat().st_mtime >= src.stat().st_mtime:
            skipped += 1
            continue
        try:
            df = loader.load_one(code, a.start, a.end, src_dir)
            if df is None:                # 区间内无行 = 未上市 / 整段停牌（业务现象，不算失败）
                empty += 1
                continue
            out, reps = pipeline.run_one(
                df, {"universe": "stock", "asset_type": "stock", "code": code})
            if out is None or len(out) == 0:
                raise ValueError("清洗后是空表（不许写空文件）")
            agg["bytes"] += store_one(out, dest, code)
            agg["rows"] += len(out)
            agg["stocks"] += 1
            written.append(code)
            for r in reps:
                cols[r["name"]].update(r["columns_added"])
                agg["warnings"] += len(r["warnings"])
            for stage, key in AGG_FIELDS:
                agg[key] += int(out.attrs.get(stage, {}).get(key, 0))
        except Exception as e:                          # noqa: BLE001 —— 单只失败不中断全库
            failures.append({"code": code, "error": f"{type(e).__name__}: {e}"})
        if not a.quiet and (i <= 2 or i % a.every == 0 or i == len(codes)):
            dt = time.time() - t0
            print(f"[{i:>5}/{len(codes)}] 已写 {agg['stocks']:>5} 只 {agg['rows']:>10,} 行 "
                  f"{agg['bytes'] / 2**20:6.0f} MB  用时 {dt:6.1f}s  速度 {i / max(dt, 1e-9):5.1f} 只/秒")

    elapsed = time.time() - t0
    src_files = sorted(src_dir.glob("*.parquet"))
    src_set = {p.stem for p in src_files}
    orphans = sorted(p.stem for p in dest.glob("*.parquet") if p.stem not in src_set)
    tmps = sorted(p.name for p in dest.glob("*.parquet.tmp"))

    meta_p.write_text(json.dumps({
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "interval": [a.start, a.end],
        "pipeline": [n for n, _ in pipeline.PIPELINE],
        "out_dir": str(dest),
        "stocks": agg["stocks"], "rows": agg["rows"], "bytes": agg["bytes"],
        "agg": {key: int(agg[key]) for _, key in AGG_FIELDS} | {"warnings": int(agg["warnings"])},
        "columns_added": {n: sorted(cols[n]) for n in STAGES},
        "failures": failures, "skipped": skipped, "empty_interval": empty,
        "orphans": orphans, "written": written,
        "elapsed_s": round(elapsed, 1),
        "source": source_fingerprint(src_files),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 100)
    print(f"合计：{agg['stocks']:,} 只 | {agg['rows']:,} 行 | {agg['bytes'] / 2**20:.0f} MB | "
          f"用时 {elapsed:.0f}s（{agg['stocks'] / max(elapsed, 1e-9):.1f} 只/秒）")
    print("各环节新增列：" + " ｜ ".join(f"{n}{sorted(cols[n])}" for n in STAGES))
    print("attrs 汇总：" + "  ".join(f"{key}={int(agg[key]):,}" for _, key in AGG_FIELDS)
          + f"  warnings={int(agg['warnings'])}")
    if skipped:
        print(f"跳过 {skipped} 只（--skip-existing）")
    if empty:
        print(f"区间内无数据 {empty} 只（未上市 / 整段停牌，不算失败）")
    if orphans:
        print(f"⚠️ 目标目录有 {len(orphans)} 个孤儿文件（源数据里没有）：{orphans[:5]}")
    if tmps:
        print(f"⚠️ 有 {len(tmps)} 个 .tmp 残留（上次被打断，可删）：{tmps[:5]}")
    print(f"总账已写：{meta_p}")
    if failures:
        print(f"❌ 失败 {len(failures)} 只：{failures[:5]}")
        sys.exit(1)
    print("✅ 落盘完成，无失败")


if __name__ == "__main__":
    main()
