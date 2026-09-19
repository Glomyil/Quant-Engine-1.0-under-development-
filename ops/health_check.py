"""health_check.py —— 全库数据体检（只读，绝不修改任何数据）

用途：把"清洗清单是否完整"从记忆变成可执行。
判据：跑完看 UNKNOWN —— 每一个命中都必须有归属（一个事项编号，或显式标成"未登记"）。
      UNKNOWN = 0 时，才能说"在现有数据源下做到了最大限度干净"。

对应文档：
    教学文档/数据契约.md      —— 列契约（每列的语义与已知异常）
    教学文档/数据清洗事项总表.md —— 事项编号（A/D/C/E 系列）

跑法：
    python -m ops.health_check              # 全库（约 1~2 分钟）
    python -m ops.health_check --limit 200  # 只查前 200 个文件（快速冒烟）
    python -m ops.health_check --strict     # 有 UNKNOWN 时退出码 1（供将来 CI 用）
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.constants import STOCK_DAILY_DIR
from data.cleaner.base import COLUMN_WHITELIST, REQUIRED_COLUMNS

# (编号, 归属事项, 名称, 类型)  类型: 异常 / 信息
# 归属写 "未登记" 的，就是当前清单的缺口 —— 体检报告会把它算进 UNKNOWN
CHECKS = [
    ("C01", "A4", "日期无法解析", "异常"),
    ("C02", "A5", "日期重复", "异常"),
    ("C03", "A6", "日期非升序", "异常"),
    ("C04", "A1/A2", "缺必需列", "异常"),
    ("C05", "A1/A2", "空表（0 行）", "异常"),
    ("C06", "D1", "OHLC 关系违例", "异常"),
    ("C07", "D2", "价格 NaN（任一路价格）", "异常"),
    ("C08", "D2", "非正价格（<=0）", "异常"),
    ("C09", "D2", "负成交量 / 负成交额", "异常"),
    ("C10", "未登记", "volume>0 而 amount=0（跨列一致性）", "异常"),
    ("C11", "未登记", "amount/volume 不在 [low,high]（物理校验）", "异常"),
    ("C12", "未登记", "复权因子非正 / 非单调（时序）", "异常"),
    ("C13", "E5", "tradestatus 非法取值", "异常"),
    ("C14", "C1", "停牌行数（信息）", "信息"),
    ("C15", "C1", "停牌行且价格全缺（第 5 形态）", "异常"),
    ("C16", "C1", "停牌行 raw 价与前收不等（多为停牌期间除权）", "信息"),
    ("C20", "未登记", "停牌行后复权价与前一日不等（真·价格不变量违约）", "异常"),
    ("C21", "未登记", "复牌日不变量违约（is_resume=1 但 resume_after_days<1）", "异常"),
    ("C17", "E5", "isST 非法取值", "异常"),
    ("C18", "—", "code 列与文件名不一致", "异常"),
    ("C19", "—", "出现未登记的新列（schema 漂移）", "信息"),
]
META = {c[0]: c for c in CHECKS}
REGISTERED_COLS = set(REQUIRED_COLUMNS) | set(COLUMN_WHITELIST)


def audit_file(path: Path, df: pd.DataFrame, show: int, details: dict, state: dict) -> dict:
    n = len(df)
    c = {cid: 0 for cid in META}
    code = path.stem

    def rec(cid, mask_or_n, note=""):
        if isinstance(mask_or_n, (int, np.integer)):
            k = int(mask_or_n)
            idx = []
        else:
            m = mask_or_n
            k = int(m.sum())
            idx = list(df.index[m])[:show] if k and show else []
        if k:
            c[cid] = k
            for i in idx:
                details.setdefault(cid, []).append(
                    f"{code} {str(df['date'].iloc[i])[:10]} {note}")
        return k

    missing = [x for x in REQUIRED_COLUMNS if x not in df.columns]
    if missing:
        c["C04"] = len(missing)
        details.setdefault("C04", []).append(f"{code} 缺 {missing}")
    extra = [x for x in df.columns if x not in REGISTERED_COLS]
    if extra:
        c["C19"] = 1
        state.setdefault("extra_cols", set()).update(extra)
    if n == 0:
        c["C05"] = 1
        details.setdefault("C05", []).append(f"{code} 空表")
        return c
    if missing:
        return c

    d = pd.to_datetime(df["date"], errors="coerce")
    rec("C01", d.isna())
    rec("C02", d.duplicated())
    rec("C03", 0 if d.is_monotonic_increasing else 1, "非升序")

    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    lo = pd.to_numeric(df["low"], errors="coerce")
    cl = pd.to_numeric(df["close"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")
    a = pd.to_numeric(df["amount"], errors="coerce")

    rec("C07", o.isna() | h.isna() | lo.isna() | cl.isna())
    rec("C08", (o <= 0) | (h <= 0) | (lo <= 0) | (cl <= 0))
    rec("C06", (h < np.maximum(o, cl) - 1e-6) | (lo > np.minimum(o, cl) + 1e-6) | (h < lo - 1e-6))
    rec("C09", (v < 0) | (a < 0))
    rec("C10", (v > 0) & (a == 0))
    good = (v > 0) & (a > 0) & cl.notna() & lo.notna() & h.notna()
    if good.any():
        vw = a[good] / v[good]
        out = (vw < lo[good] * 0.98) | (vw > h[good] * 1.02)
        idx = list(df.index[good][out])[:show] if show else []
        if int(out.sum()):
            c["C11"] = int(out.sum())
            for i in idx:
                details.setdefault("C11", []).append(
                    f"{code} {str(df['date'].iloc[i])[:10]} vwap={float(a[i] / v[i]):.4f} "
                    f"区间[{float(lo[i]):.2f},{float(h[i]):.2f}]")

    if "back_adj_factor" in df.columns:
        f = pd.to_numeric(df["back_adj_factor"], errors="coerce")
        bad = f.isna() | (f <= 0)
        nonmono = 0 if f.is_monotonic_increasing else 1
        if int(bad.sum()) or nonmono:
            c["C12"] = int(bad.sum()) + nonmono
            details.setdefault("C12", []).append(
                f"{code} 非正{int(bad.sum())}个 单调={f.is_monotonic_increasing}")

    if "tradestatus" in df.columns:
        ts = pd.to_numeric(df["tradestatus"], errors="coerce")
        rec("C13", ~ts.isin([0, 1]))
        susp = ts == 0
        rec("C14", susp, "停牌")
        rec("C15", susp & cl.isna())
        prev = cl.shift(1)
        fac = (pd.to_numeric(df["back_adj_factor"], errors="coerce")
               if "back_adj_factor" in df.columns else pd.Series(1.0, index=df.index))
        hfq = cl * fac
        hfq_prev = hfq.shift(1)
        m16 = susp & cl.notna() & prev.notna() & ~np.isclose(cl, prev, rtol=0, atol=1e-6)
        if int(m16.sum()):
            c["C16"] = int(m16.sum())
        m20 = susp & hfq.notna() & hfq_prev.notna() & ~np.isclose(hfq, hfq_prev, rtol=1e-6, atol=1e-4)
        k20 = int(m20.sum())
        if k20:
            c["C20"] = k20
            for i in list(df.index[m20])[:show]:
                details.setdefault("C20", []).append(
                    f"{code} {str(df['date'].iloc[i])[:10]} hfq={float(hfq[i]):.4f} "
                    f"前一日hfq={float(hfq_prev[i]):.4f} close={float(cl[i]):.2f} 前收={float(prev[i]):.2f}")

    if "isST" in df.columns:
        st = pd.to_numeric(df["isST"], errors="coerce")
        rec("C17", ~st.isin([0, 1]))

    # C21：复牌日不变量（只在清洗后数据上成立；原始 parquet 没有这两列 → 天然跳过）
    if "is_resume" in df.columns and "resume_after_days" in df.columns:
        ir = pd.to_numeric(df["is_resume"], errors="coerce")
        rad = pd.to_numeric(df["resume_after_days"], errors="coerce")
        rec("C21", (ir == 1) & (rad < 1))

    if "code" in df.columns:
        rec("C18", df["code"].astype(str) != code)

    return c


def main() -> int:
    ap = argparse.ArgumentParser(description="全库数据体检（只读）")
    ap.add_argument("--limit", type=int, default=0, help="只查前 N 个文件（0=全库）")
    ap.add_argument("--show", type=int, default=3, help="每类最多列几条明细")
    ap.add_argument("--strict", action="store_true", help="有 UNKNOWN 时退出码 1")
    args = ap.parse_args()

    files = sorted(Path(STOCK_DAILY_DIR).glob("*.parquet"))
    if args.limit:
        files = files[: args.limit]
    print(f"体检目标: {STOCK_DAILY_DIR}")
    print(f"文件数: {len(files)}")

    totals = {cid: 0 for cid in META}
    details: dict = {}
    state: dict = {}
    rows = 0
    t0 = time.time()
    for i, p in enumerate(files, 1):
        try:
            df = pd.read_parquet(p)
        except Exception as e:  # 读不了也要记账，不能静默
            totals["C04"] += 1
            details.setdefault("C04", []).append(f"{p.stem} 读取失败: {type(e).__name__}")
            continue
        rows += len(df)
        for cid, k in audit_file(p, df, args.show, details, state).items():
            totals[cid] += k
        if i % 500 == 0:
            print(f"  ... {i}/{len(files)}  ({time.time() - t0:.0f}s)", flush=True)

    dur = time.time() - t0
    print("")
    print("=" * 78)
    print(f"行数合计: {rows:,}   用时: {dur:.1f}s")
    print("=" * 78)
    print(f"{'编号':<5}{'命中':>10}  {'归属':<8}  检查项")
    print("-" * 78)
    for cid, owner, name, kind in CHECKS:
        print(f"{cid:<5}{totals[cid]:>10}  {owner:<8}  {name}{'  [信息]' if kind == '信息' else ''}")

    unk = [(cid, totals[cid]) for cid, owner, _, kind in CHECKS
           if owner == "未登记" and totals[cid] > 0]
    hit_abn = [(cid, totals[cid]) for cid, owner, _, kind in CHECKS
               if kind == "异常" and totals[cid] > 0]

    print("")
    print("== 明细（每类最多 %d 条）==" % args.show)
    shown = False
    for cid, owner, name, kind in CHECKS:
        if cid in details and totals[cid]:
            shown = True
            print(f"[{cid} {name}]")
            for line in details[cid][: args.show + 5]:
                print("   ", line)
    if not shown:
        print("    （无）")

    if state.get("extra_cols"):
        print("")
        print("未登记列名（上游产出，白名单只管环节新增列）: "
              + ", ".join(sorted(state["extra_cols"])))
    print("")
    print("== 结论 ==")
    print(f"命中异常的检查类: {len(hit_abn)} 类 -> {[c for c, _ in hit_abn]}")
    if unk:
        print(f"UNKNOWN（无归属，需要补进事项总表）: {len(unk)} 类 / 共 {sum(k for _, k in unk)} 命中")
        for cid, k in unk:
            print(f"   {cid} {META[cid][2]}: {k}")
        print("→ 判据未达成：数据还没到'最大限度干净'")
    else:
        print("UNKNOWN = 0 → 所有命中都有归属，判据达成 ✅")
    print("")
    print("提示：C19（未登记列）是信息型 —— 它报告的是上游产出的列没写进白名单，")
    print("      不代表错误（白名单只管'环节新增的列'），但 schema 变了要记账。")
    return 1 if (args.strict and unk) else 0


if __name__ == "__main__":
    raise SystemExit(main())
