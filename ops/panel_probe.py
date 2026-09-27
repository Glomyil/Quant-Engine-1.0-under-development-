"""ops/panel_probe.py —— 独立探针：自己算一遍面板统计，给 panel.py 的验收提供参照值

用途：规格《教学文档/panel.py教学参考.md》§3.2 要"全库对账"，但那些数字是 **hfq_open 落盘之前** 量的。
      本探针**不 import backtest/panel.py**，自己走一遍 cleaner_daily → 4 张宽表，
      给出形状 / 覆盖率 / 可交易标的数 / dtype / 内存 / 缓存体积 / 读写耗时 —— 两条独立路径对上了才算数。

不做：不写 backtest/cache/（那是 panel.py 的产物）；本探针的临时产物落在 data/storage/_probe_panel/。

跑法（项目根目录）：
    python -m ops.panel_probe
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from config import constants as C

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = Path(C.CLEANER_DAILY_DIR)
OUT = C.DATA_STORAGE / "_probe_panel"
FIELDS = ["hfq_close", "hfq_open", "ret_1d_hfq", "can_trade"]


def mb(df_or_n) -> float:
    return round(float(df_or_n) / 2**20, 1)


def main() -> None:
    files = sorted(SRC.glob("*.parquet"))
    print(f"源 {SRC}  文件 {len(files):,} 只")
    t0 = time.time()
    c_l, o_l, r_l, t_l = [], [], [], []
    nan_close = nan_open = nan_ret = 0
    for i, p in enumerate(files, 1):
        d = pd.read_parquet(p)
        idx = pd.DatetimeIndex(pd.to_datetime(d["date"]))
        c = pd.to_numeric(d["hfq_close"], errors="coerce").to_numpy()
        nan_close += int(np.isnan(c).sum())
        nan_open += int(pd.to_numeric(d["hfq_open"], errors="coerce").isna().sum())
        nan_ret += int(pd.to_numeric(d["ret_1d_hfq"], errors="coerce").isna().sum())
        c_l.append(pd.Series(c, index=idx, name=p.stem))
        o_l.append(pd.Series(pd.to_numeric(d["hfq_open"], errors="coerce").to_numpy(), index=idx, name=p.stem))
        r_l.append(pd.Series(pd.to_numeric(d["ret_1d_hfq"], errors="coerce").to_numpy(), index=idx, name=p.stem))
        ok = ((pd.to_numeric(d["is_suspended"], errors="coerce").fillna(1).to_numpy() == 0)
              & (pd.to_numeric(d["is_price_bad"], errors="coerce").fillna(1).to_numpy() == 0)
              & pd.notna(c))
        t_l.append(pd.Series(ok, index=idx, name=p.stem))
        if i <= 2 or i % 2000 == 0:
            print(f"  [{i:>5}/{len(files)}] {p.stem}  {time.time() - t0:6.1f}s")
    build_s = time.time() - t0

    panels = {"hfq_close": pd.concat(c_l, axis=1).sort_index().astype("float32"),
              "hfq_open": pd.concat(o_l, axis=1).sort_index().astype("float32"),
              "ret_1d_hfq": pd.concat(r_l, axis=1).sort_index().astype("float32"),
              "can_trade": pd.concat(t_l, axis=1).sort_index().eq(True)}
    del c_l, o_l, r_l, t_l

    shape = panels["hfq_close"].shape
    cov = float(panels["hfq_close"].notna().to_numpy().mean())
    per_day = panels["can_trade"].sum(axis=1)
    print("\n========== 面板（现算，内存） ==========")
    print(f"装配耗时            {build_s:.0f} s")
    print(f"形状                {shape}")
    print(f"hfq_close 覆盖率    {cov:.4%}")
    print(f"长表内 NaN 行      hfq_close {nan_close:,} ｜ hfq_open {nan_open:,} ｜ ret_1d_hfq {nan_ret:,}"
          f"（对账 adjust / derived 的 attrs 记账）")
    print(f"面板 NaN 格        hfq_close {int(panels['hfq_close'].isna().to_numpy().sum()):,}"
          f"（结构性稀疏 = 未上市 / 已退市 / 停牌缺行，不是数据缺陷）")
    print(f"可交易标的 中位/最少/最多  {int(per_day.median())} / {int(per_day.min())} / {int(per_day.max())}")
    print(f"index 升序唯一      {panels['hfq_close'].index.is_monotonic_increasing and panels['hfq_close'].index.is_unique}")
    print(f"日期范围            {panels['hfq_close'].index[0].date()} ~ {panels['hfq_close'].index[-1].date()}")
    for k, v in panels.items():
        print(f"  {k:<11} dtype {str(v.dtypes.iloc[0]):<8} 内存 {mb(v.memory_usage(deep=True).sum())} MB")

    print("\n========== 落临时盘（口径同 panel.py，但目录是 _probe_panel） ==========")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    for k, v in panels.items():
        v.to_parquet(OUT / f"probe_{k}.parquet")
    write_s = time.time() - t1
    size = sum(p.stat().st_size for p in OUT.glob("*.parquet"))
    print(f"写盘 {write_s:.1f} s   体积 {mb(size)} MB（float32）")
    t2 = time.time()
    back = {k: pd.read_parquet(OUT / f"probe_{k}.parquet") for k in FIELDS}
    read_s = time.time() - t2
    print(f"读回 {read_s:.2f} s   逐表相等：", end="")
    same = all(back[k].equals(panels[k]) for k in FIELDS)
    print("全部相等 ✅" if same else "❌ 有不一致")
    print(f"单张最大 {mb(max(p.stat().st_size for p in OUT.glob('*.parquet')))} MB"
          f"（{max(OUT.glob('*.parquet'), key=lambda p: p.stat().st_size).name}）")

    (OUT.parent / "_probe_panel_meta.json").write_text(json.dumps({
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "shape": list(shape),
        "coverage_hfq_close": round(cov, 4),
        "tradable": [int(per_day.median()), int(per_day.min()), int(per_day.max())],
        "build_s": round(build_s, 1), "write_s": round(write_s, 1), "read_s": round(read_s, 2),
        "bytes": int(size), "source_files": len(files),
        "nan_rows": {"hfq_close": nan_close, "hfq_open": nan_open, "ret_1d_hfq": nan_ret},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n参照值已写：{OUT.parent / '_probe_panel_meta.json'}")
    print("（临时产物，核对完可整个删掉：data/storage/_probe_panel/ 与 _probe_panel_meta.json）")


if __name__ == "__main__":
    main()
