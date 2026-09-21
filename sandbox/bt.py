"""sandbox/bt.py —— 最小闭环：数据 → 因子 → 权重 → 撮合 → 指标 → 报告

一句话：把 5,471 个 parquet 跑成一份"分层净值 + 指标 + 图 + 报告"。
设计原则：先跑通、再跑对、最后跑好 —— 故意不做的东西全部写在报告的"局限"里。

跑法（项目根目录）：
    python -m sandbox.bt
改参数：只改下面"可调参数"那 8 行。
产物：sandbox/out/ 下 report_<因子>.md / nav_<因子>.png / trades_<因子>.parquet / all_factors.md
"""
from __future__ import annotations
from sandbox import factors as F
from data import loader
import pandas as pd
import numpy as np

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ========== 可调参数（只改这里）==========================================
MODE = "raw"                  # raw = 直接用原始 parquet（全库约 37 秒）| cleaned = 走你的清洗环节
FACTOR_LIST = ["rev_5d", "rev_20d", "mom_120d", "vol_20d", "ma_dev_20", "amihud_20"]
N_CODES = None                # None = 全库；调试时填 50
START, END = "2015-01-01", "2026-12-31"
N_LAYERS, REBAL, FEE = 5, "ME", 0.0005
OUT = _ROOT / "sandbox" / "out"
CLEAN_OUT = True              # 每次重跑前，先清掉上一次的产物（报告/图/净值/账本）
KEEP_PANEL_CACHE = True       # True = 保留面板缓存（省 40 秒）｜False = 连缓存一起清
# ========================================================================

TRADE_COLUMNS = ["date", "code", "layer", "side",
                 "price", "shares", "amount", "fee", "reason"]


# -- (0) 重跑前清场：只清"我们自己产生的产物"，不碰缓存和别人的文件 ---------
def clean_out() -> int:
    """删掉上一次的 report_*/nav_*/trades_*/all_factors.md，返回删除个数

    设计取舍：面板缓存（panel_*.parquet，全库约 258 MB / 建一次 40~60 秒）
    默认**保留** —— 它是"缓存"不是"结果"，每次重建纯属浪费。
    想连缓存一起清：把顶部 KEEP_PANEL_CACHE 改成 False。
    ⚠️ 只按固定文件名模式删，绝不用通配符清空整个目录（防误删你放进来的东西）。
    """
    patterns = ["report_*.md", "nav_*.parquet", "nav_*.png", "trades_*.parquet", "all_factors.md"]
    if not KEEP_PANEL_CACHE:
        patterns.append("panel_*.parquet")
    n = 0
    for pat in patterns:
        for f in OUT.glob(pat):
            f.unlink()
            n += 1
    return n


# -- (1) 数据装配：长表 -> 宽表（两种模式，同一个契约）-------------------
def load_panels(codes, mode=MODE, start=START, end=END) -> dict:
    """返回 {close_adj, open_adj, amount, can_trade}；index=交易日, columns=代码

    raw     ：只用原始列 —— 复权价 = close x back_adj_factor；可交易 = 非停牌 且 非 ST 且 有价
    cleaned ：先过你的清洗环节（align -> suspension -> price），再叠清洗后的标记
    """
    tag = f"{mode}_{len(codes)}_{start[:4]}_{end[:4]}"
    paths = {k: OUT / f"panel_{tag}_{k}.parquet"
             for k in ("close_adj", "open_adj", "amount", "can_trade")}
    if all(p.exists() for p in paths.values()):
        return {k: pd.read_parquet(p) for k, p in paths.items()}
    steps = None
    if mode == "cleaned":
        from data.cleaner import align, price, pipeline, suspension
        steps = [("align", align), ("suspension",
                                    suspension), ("price", price)]
    c_l, o_l, a_l, t_l = [], [], [], []
    for code, df in loader.iter_frames(codes, start, end):
        if steps:
            df = pipeline.run_one(df, {"universe": "stock", "asset_type": "stock", "code": code},
                                  steps=steps)[0]
        d = df.sort_values("date").drop_duplicates(
            "date", keep="first").reset_index(drop=True)
        idx = pd.DatetimeIndex(pd.to_datetime(d["date"]))
        f = pd.to_numeric(d["back_adj_factor"], errors="coerce").to_numpy()
        o_raw = pd.to_numeric(d["open"], errors="coerce").to_numpy()
        c_l.append(pd.Series(pd.to_numeric(d["close"], errors="coerce").to_numpy() * f,
                             index=idx, name=code))
        o_l.append(pd.Series(o_raw * f, index=idx, name=code))
        a_l.append(pd.Series(pd.to_numeric(d["amount"], errors="coerce").to_numpy(),
                             index=idx, name=code))
        ts = pd.to_numeric(d["tradestatus"], errors="coerce").to_numpy()
        st = pd.to_numeric(d["isST"], errors="coerce").to_numpy()
        ok = (ts != 0) & (st != 1) & pd.notna(o_raw)
        if steps and "is_suspended" in d.columns:
            ok = ok & (d["is_suspended"].to_numpy() == 0)
        if steps and "is_price_bad" in d.columns:
            ok = ok & (d["is_price_bad"].to_numpy() == 0)
        t_l.append(pd.Series(ok, index=idx, name=code))
    panels = {"close_adj": pd.concat(c_l, axis=1).sort_index(),
              "open_adj": pd.concat(o_l, axis=1).sort_index(),
              "amount": pd.concat(a_l, axis=1).sort_index(),
              "can_trade": pd.concat(t_l, axis=1).sort_index().astype(bool)}
    OUT.mkdir(parents=True, exist_ok=True)
    for k, p in paths.items():
        panels[k].to_parquet(p)          # parquet 已被 .gitignore 挡掉，不会污染版本库
    return panels


# -- (2) 因子：从 registry 取，检查依赖 + sign 定向 ----------------------
def get_factor(name, panels, params=None) -> pd.DataFrame:
    """返回定向后的因子表：数值越大 = 越看多下一期（层4 = 看多档）"""
    spec = F.FACTORS[name]
    miss = [k for k in spec["requires"] if k not in panels]
    if miss:
        raise KeyError(f"因子 {name} 需要面板里的 {miss}")
    return (spec["fn"](panels, params or {}) * spec["sign"]).astype("float64")


# -- (3) 调仓日 + 目标权重 ----------------------------------------------
def get_rebalance_dates(dates, freq=REBAL) -> pd.DatetimeIndex:
    """每个调仓周期的最后一个**真实交易日**（不是自然月末 31 号）"""
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.resample(freq).last().dropna().to_numpy())


def build_weights(factor, rdates, can_trade, n_layers=N_LAYERS) -> dict:
    """每个调仓日：因子截面排名 -> 分 n 层 -> 层内等权；层0=看空档, 层4=看多档"""
    out = {}
    for L in range(n_layers):
        w = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
        for d in rdates:
            row = factor.loc[d]
            ok = row.notna() & can_trade.loc[d].reindex(
                row.index).fillna(False)
            if not ok.any():
                continue
            r = row[ok].rank(pct=True)
            sel = r[(r > L / n_layers) & (r <= (L + 1) / n_layers)].index
            if len(sel):
                w.loc[d, sel] = 1.0 / len(sel)
        out[L] = w
    return out


# -- (4) 撮合与净值：T+1 开盘成交 + 现金约束 + 估值 ffill -----------------
def simulate(weights, panels, layer):
    """返回 (每日净值, 账本, 每期双边换手)。信号日 t 算权重 -> t+1 开盘成交"""
    open_adj, close_adj, can_trade = panels["open_adj"], panels["close_adj"], panels["can_trade"]
    idx = close_adj.index
    px_val = close_adj.ffill()                      # 估值用（停牌期间沿用最后价）
    shares = pd.Series(0.0, index=close_adj.columns)
    cash, nav, recs, turn = 1.0, pd.Series(np.nan, index=idx), [], {}
    pending = None
    for d in idx:
        if pending is not None:
            px_exec = open_adj.loc[d].where(
                can_trade.loc[d])       # 不能交易 -> NaN
            val_px = px_exec.fillna(px_val.loc[d])
            pos_val = (shares * val_px).fillna(0.0)
            total = cash + float(pos_val.sum())
            delta = (pending * total - pos_val).where(px_exec.notna(), 0.0)
            sell_sh = ((-delta).clip(lower=0) /
                       px_exec).where(px_exec.notna()).fillna(0.0)
            sell_sh = sell_sh.clip(upper=shares.clip(lower=0))
            proceeds = float((sell_sh * px_exec).sum())
            for cd in sell_sh[sell_sh > 0].index:
                amt = float(sell_sh[cd] * px_exec[cd])
                recs.append((d, cd, layer, "sell", float(px_exec[cd]),
                             float(sell_sh[cd]), amt, amt * FEE, "调仓"))
            shares -= sell_sh
            cash += proceeds - proceeds * FEE
            buys = delta.clip(lower=0)
            cost = 0.0
            if buys.sum() > 0 and cash > 0:
                scale = min(1.0, cash / (buys.sum() * (1 + FEE)))
                buy_sh = (buys * scale / px_exec).fillna(0.0)
                cost = float((buys * scale).sum() * (1 + FEE))
                for cd in buy_sh[buy_sh > 0].index:
                    amt = float(buy_sh[cd] * px_exec[cd])
                    recs.append((d, cd, layer, "buy", float(px_exec[cd]),
                                 float(buy_sh[cd]), amt, amt * FEE, "调仓"))
                shares += buy_sh
                cash -= cost
            turn[d] = (proceeds + cost) / max(total, 1e-12)
            pending = None
        nav[d] = cash + float((shares * px_val.loc[d]).fillna(0.0).sum())
        wr = weights.loc[d]
        if wr.notna().any():
            pending = wr.fillna(0.0)                # 明天执行（T+1）
    return nav, pd.DataFrame(recs, columns=TRADE_COLUMNS), pd.Series(turn)


# -- (5) 指标：年化/波动/夏普/回撤/月胜率/换手/IC/ICIR -------------------
def compute_metrics(nav, trades, factor, close_adj, rdates) -> dict:
    r = nav.pct_change().dropna()
    ann = nav.iloc[-1] ** (252 / max(len(nav), 1)) - 1
    vol = r.std() * np.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()
    mon = nav.resample("ME").last().pct_change().dropna()
    px = close_adj.ffill()
    ics = []
    for i in range(len(rdates) - 1):
        d0, d1 = rdates[i], rdates[i + 1]
        pair = pd.concat([factor.loc[d0], px.loc[d1] /
                         px.loc[d0] - 1], axis=1).dropna()
        if len(pair) > 20:
            ics.append(pair.iloc[:, 0].corr(
                pair.iloc[:, 1], method="spearman"))
    ics = pd.Series(ics)
    return {"年化": ann, "年化波动": vol, "夏普": (r.mean() * 252) / vol if vol > 0 else np.nan,
            "最大回撤": mdd, "月胜率": (mon > 0).mean(), "换手": np.nan,
            "IC均值": ics.mean(), "IC标准差": ics.std(),
            "ICIR": ics.mean() / ics.std() if ics.std() > 0 else np.nan,
            "IC t值": ics.mean() / (ics.std() / np.sqrt(len(ics))) if len(ics) > 1 else np.nan,
            "IC期数": len(ics), "成交笔数": len(trades)}


# -- (6) 报告 + 图 -------------------------------------------------------
def _table(head, rows) -> str:
    out = ["| " + " | ".join(head) + " |", "| " +
           " | ".join([":--"] * len(head)) + " |"]
    out += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return chr(10).join(out)


def write_report(name, navs, mets, mls, trades, mode, n_codes) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"layer{k}": v for k, v in navs.items()}
                 ).to_parquet(OUT / f"nav_{name}.parquet")
    trades.to_parquet(OUT / f"trades_{name}.parquet")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        for L, nav in navs.items():
            ax.plot(nav.index, nav.values, lw=1.1, label=f"layer{L}")
        ax.plot(navs[0].index, (navs[0] / navs[4]).values, ls="--", c="k", lw=1.4,
                label="long-short(L0/L4)")
        ax.set_title(f"{name} layered NAV | mode={mode} | monthly, T+1 open")
        ax.legend(ncol=3)
        fig.tight_layout()
        fig.savefig(OUT / f"nav_{name}.png", dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f"   (plot skipped: {type(e).__name__})")
    head = ["组合", "年化", "年化波动", "夏普", "最大回撤", "月胜率", "单边换手/月"]
    rows = []
    for L in range(len(navs)):
        m = mets[L]
        rows.append([f"层{L}", f"{m['年化']*100:.2f}%", f"{m['年化波动']*100:.2f}%",
                     f"{m['夏普']:.2f}", f"{m['最大回撤']*100:.2f}%",
                     f"{m['月胜率']*100:.0f}%", f"{m['换手']*100:.1f}%"])
    rows.append([f"多空(层{N_LAYERS-1}/层0)", f"{mls['年化']*100:.2f}%", f"{mls['年化波动']*100:.2f}%",
                 f"{mls['夏普']:.2f}", f"{mls['最大回撤']*100:.2f}%",
                 f"{mls['月胜率']*100:.0f}%", "-"])
    m0 = mets[0]
    md = f"""# {name} 分层回测（最小闭环 v0）

数据模式 **{mode}** | **{n_codes} 只标的** | 样本 {START} ~ {END} | 因子 {name}（{F.FACTORS[name]['note']}，sign={F.FACTORS[name]['sign']}）

## 结论
层4（看多档）年化 {mets[4]['年化']*100:.2f}%，层0（看空档）年化 {mets[0]['年化']*100:.2f}%；
多空（买层{N_LAYERS-1}、卖层0）年化 {mls['年化']*100:.2f}%、夏普 {mls['夏普']:.2f}、最大回撤 {mls['最大回撤']*100:.2f}%。
因子侧：IC 均值 {m0['IC均值']:+.4f}、ICIR {m0['ICIR']:+.3f}（{m0['IC期数']} 期）。

## 指标
{_table(head, rows)}

## 局限（v0 故意不做的）
1. 不处理一字板/涨跌停：价格触限时仍算成交，会略微高估收益
2. 不计滑点与冲击成本，只有单边 {FEE*100:.2f}% 固定费率
3. 退市股按最后有效价持有到数据结束，不做强制平仓
4. 5 层等权，不做市值/行业中性化
5. 夏普口径：rf=0、年化 252 天；多空为 nav(层0)/nav(层4) 近似
6. 未做多重检验校正；未切样本外

## 下一步
换因子窗口 / 换调仓频率 / 加上 ST 与涨跌停约束 / 切样本外验证
"""
    (OUT / f"report_{name}.md").write_text(md, encoding="utf-8")
    print("   -> " + str(OUT / ("report_" + name + ".md")))


def write_compare(results, mode, n_codes) -> None:
    head = ["因子", "sign", "层4年化", "多空年化", "多空夏普", "多空MDD",
            "IC均值", "ICIR", "IC t值", "单边换手/月"]
    rows = []
    for name, r in results.items():
        m4, mls, m0 = r["mets"][4], r["ls"], r["mets"][0]
        rows.append([name, F.FACTORS[name]["sign"], f"{m4['年化']*100:.2f}%",
                     f"{mls['年化']*100:.2f}%", f"{mls['夏普']:.2f}",
                     f"{mls['最大回撤']*100:.2f}%", f"{m0['IC均值']:+.4f}",
                     f"{m0['ICIR']:+.3f}", f"{m0['IC t值']:+.2f}", f"{m4['换手']*100:.1f}%"])
    rows.sort(key=lambda x: -float(x[4]))
    md = f"""# 多因子对比（最小闭环 v0）

数据模式 **{mode}** | **{n_codes} 只标的** | {START} ~ {END} | 月频调仓 / T+1 开盘成交 / 单边费 {FEE*100:.2f}%
/ 5 层等权 / 全 A 含退市

{_table(head, rows)}

## 口径与纪律
- 本次一共测了 **{len(results)}** 个因子（测得多总有好看的，读者有权知道分母）
- 未做多重检验校正；未切样本外（下一步：2015-2020 挑、2021-2026 验）
"""
    (OUT / "all_factors.md").write_text(md, encoding="utf-8")
    print("   -> " + str(OUT / "all_factors.md"))


# -- (7) 入口 ------------------------------------------------------------
def main():
    t0 = time.time()
    if CLEAN_OUT:
        print(f"(0) 清场：删掉上一次 {clean_out()} 个产物文件"
              f"（面板缓存{'保留' if KEEP_PANEL_CACHE else '也清掉'}）")
    codes = loader.list_codes()
    if N_CODES:
        codes = codes[:N_CODES]
    print(f"模式 {MODE} | 标的 {len(codes)} 只 | 因子 {FACTOR_LIST}")
    panels = load_panels(codes)
    print(f"(1) 面板 {panels['close_adj'].shape}（{time.time()-t0:.0f}s）")
    rdates = get_rebalance_dates(panels["close_adj"].index)
    print(
        f"(2) 调仓日 {len(rdates)} 个（{str(rdates[0])[:10]} ~ {str(rdates[-1])[:10]}）")
    results = {}
    for name in FACTOR_LIST:
        try:
            factor = get_factor(name, panels)
        except (KeyError, NotImplementedError) as e:
            print(f"(3) 跳过 {name}：{type(e).__name__}: {e}")
            continue
        w = build_weights(factor, rdates, panels["can_trade"])
        navs, mets, trs = {}, {}, []
        for L in range(N_LAYERS):
            nav, tr, turn = simulate(w[L], panels, L)
            navs[L] = nav
            trs.append(tr)
            mets[L] = compute_metrics(
                nav, tr, factor, panels["close_adj"], rdates)
            mets[L]["换手"] = float(turn.mean()) / 2
        mls = compute_metrics(navs[N_LAYERS - 1] / navs[0], pd.DataFrame(), factor,
                              panels["close_adj"], rdates)   # 多空 = 买"看多档"卖"看空档"
        print(f"(3) {name}: 多空夏普 {mls['夏普']:.2f} | IC {mets[0]['IC均值']:+.4f}"
              f" | ICIR {mets[0]['ICIR']:+.3f}")
        write_report(name, navs, mets, mls, pd.concat(trs), MODE, len(codes))
        results[name] = {"mets": mets, "ls": mls}
    if not results:
        print("!! 没有任何因子跑成功 —— 多半是 factors.py 里那个函数还是 TODO")
        print("   两条路：① 把它填完（通常一行）；② 把 bt.py 顶部 FACTOR_LIST 改成已实现的因子（如 rev_20d）")
        return
    write_compare(results, MODE, len(codes))
    print(f"跑完 ok 总用时 {time.time()-t0:.0f}s | 产物在 {OUT}")


if __name__ == "__main__":
    main()
