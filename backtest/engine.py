from __future__ import annotations
import numpy as np
import pandas as pd


def run_layer(W_layer, panels, rdates, *, fee_buy=0.00025, fee_sell=0.00075, slippage=0.0005, init_cash=1.0):
    days = panels["hfq_close"].index
    cols = W_layer.columns
    px_open = panels["hfq_open"][cols]
    px_val = panels["hfq_close"][cols].ffill()
    ok = panels["can_trade"][cols].eq(True)

    pos = days.get_indexer(pd.DatetimeIndex(rdates))
    exec_pos = {}
    for p, dt in zip(pos, pd.DatetimeIndex(rdates)):
        if p >= 0 and p + 1 < len(days):
            exec_pos[p+1] = dt
    exec_days = [days[i]for i in sorted(exec_pos)]

    shares = pd.Series(0.0, index=cols)
    cash = float(init_cash)
    nav = pd.Series(np.nan, index=days, dtype="float64")
    cash_s = pd.Series(np.nan, index=days, dtype="float64")
    turnover = pd.Series(0.0, index=exec_days, dtype="float64")
    fees = pd.Series(0.0, index=exec_days, dtype="float64")
    trades = pd.DataFrame(0.0, index=exec_days, columns=cols, dtype="float32")
    shares_hist = pd.DataFrame(
        0.0, index=exec_days, columns=cols, dtype="float32")

    for i, dt in enumerate(days):
        if i in exec_pos:
            w = W_layer.loc[exec_pos[i]].fillna(0.0).to_numpy()
            tradable = ok.loc[dt] & (px_open.loc[dt] > 0.0).to_numpy()
            exec_px = np.where(tradable, px_open.loc[dt].to_numpy, np.nan)
            val_px = np.nan_to_num(
                px_open.loc[dt].to_numpy().fillna(px_val.loc[dt]).to_numpy())
            nav_pre = cash + float((shares.to_numpy() * val_px).sum())
            tgt = np.where(np.isnan(exec_px), shares.to_numpy(),
                           nav_pre*w/np.nan_to_num(exec_px))
