"""
ETF 数据获取器
=============
批量下载全量 ETF 日线，分类存储为 Parquet。
数据源: 搜狐 K 线 API（免费、稳定）。

.. warning::
    **幸存者偏差**: ETF 列表来自新浪（akshare fund_etf_category_sina），
    该接口仅返回当前存续的 ETF，**已退市/清盘的 ETF 不会出现在下载列表**。
    基于此数据做的回测会高估策略表现，真实交易中退市 ETF 的亏损被排除在外。

.. warning::
    **未复权**: 搜狐 API 返回的是原始收盘价（close），非后复权价（adj_close）。
    除权除息日会出现价格跳空，导致收益率计算失真。
    对于持有期超过 1 年的策略，必须在 cleaner 模块中做复权处理。
    短期（<6 个月）策略影响有限。

工作流:
  1. fetch_all_etf_data(freq)  → 一次性全量下载，落盘 Parquet
  2. auto_update()             → 后续增量更新日线
  3. load_daily / load_minute  → 按需急速读取，按日期切片
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from config.constants import (
        DATA_STORAGE,
        ETF_DAILY_DIR,
        ETF_60MIN_DIR,
        ETF_LIST_FILE,
        ETF_META_FILE,
        MAX_RETRIES,
        RETRY_BACKOFF_BASE,
        BAO_BATCH_SIZE,
        BAO_BATCH_SLEEP,
        RANDOM_SEED,
        PARQUET_COMPRESSION,
    )
except ModuleNotFoundError:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from config.constants import (
        DATA_STORAGE, ETF_DAILY_DIR, ETF_60MIN_DIR, ETF_LIST_FILE,
        ETF_META_FILE, MAX_RETRIES, RETRY_BACKOFF_BASE, BAO_BATCH_SIZE,
        BAO_BATCH_SLEEP, RANDOM_SEED, PARQUET_COMPRESSION,
    )

np.random.seed(RANDOM_SEED)


# ============================================================================
# 模板基类
# ============================================================================

class BaseDataFeed(ABC):
    """数据源基类：所有数据源必须继承此类"""

    def __init__(self, config_path: Optional[str] = None):
        self.config: dict = self._load_config(config_path)
        self.cache: dict = {}

    @abstractmethod
    def load_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_minute(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        pass

    def _load_config(self, path: Optional[str] = None) -> dict:
        return {}

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        required: list[str] = ["date", "open",
                               "high", "low", "close", "volume"]
        return all(col in df.columns for col in required)


# ============================================================================
# AKShare ETF 数据源
# ============================================================================

class ETFDataFeed(BaseDataFeed):
    """
    ETF 全量数据源。

    频率: 日线 (d)，60分钟线暂不支持。

    存储:
      /data/storage/etf_daily/{code}.parquet
      /data/storage/etf_list.parquet
      /data/storage/etf_meta.json

    数据来源: 搜狐 K 线 API（免费、稳定），单次调用拿到 ETF 上市至今全量日线。
    ETF 列表来源: AKShare (新浪)。
    价格类型: 原始收盘价（未复权），复权处理由 cleaner 模块负责。
    """

    # ========================================================================
    # __init__
    # ========================================================================
    def __init__(self, config_path: Optional[str] = None) -> None:
        super().__init__(config_path)
        self._ensure_dirs()
        self._df_cache: Dict[str, pd.DataFrame] = {}
        self._default_start: str = self.config.get(
            "etf.start_date", "2015-01-01")
        self._default_end: str = self.config.get(
            "etf.end_date", datetime.today().strftime("%Y-%m-%d"))

    # ========================================================================
    # load_daily / load_minute
    # ========================================================================
    def load_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._fast_read(symbol, start_date, end_date, freq="d")

    def load_minute(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._fast_read(symbol, start_date, end_date, freq="60min")

    # ========================================================================
    # _load_config
    # ========================================================================
    def _load_config(self, path: Optional[str] = None) -> dict:
        config_file: Path = (
            Path(__file__).resolve().parent.parent.parent /
            "config" / "application.yml"
        )
        if path is not None:
            config_file = Path(path)
        if not config_file.exists():
            return {}
        cfg: dict = {}
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if ":" in s:
                        k, _, v = s.partition(":")
                        k, v = k.strip(), v.strip()
                        if v.lower() == "true":
                            cfg[k] = True
                        elif v.lower() == "false":
                            cfg[k] = False
                        elif v.isdigit():
                            cfg[k] = int(v)
                        else:
                            try:
                                cfg[k] = float(v)
                            except ValueError:
                                cfg[k] = v
        except Exception:
            print(f"[警告] 配置文件读取失败: {traceback.format_exc()}")
            pass
        return cfg

    # ========================================================================
    # _validate_dataframe
    # ========================================================================
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        if not super()._validate_dataframe(df):
            return False
        if df.empty:
            return False
        return True

    # ========================================================================
    # 核心业务 1：获取全量 ETF 代码列表
    # ========================================================================
    def fetch_all_etf_list(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        从 AKShare (新浪) 获取全量 ETF 代码。
        返回字段: code, name
        结果缓存在 etf_list.parquet。
        """
        if (not force_refresh) and ETF_LIST_FILE.exists():
            try:
                df = pd.read_parquet(ETF_LIST_FILE)
                if not df.empty:
                    return df
            except Exception:
                print(f"[警告] ETF列表缓存读取失败: {traceback.format_exc()}")
                pass

        import akshare as ak
        raw = ak.fund_etf_category_sina(symbol="ETF基金")

        df = pd.DataFrame({
            "code": raw.iloc[:, 0].tolist(),
            "name": raw.iloc[:, 1].tolist(),
        })
        # code 格式: sz159998 → 保留原样作为文件命名
        df = df.drop_duplicates(subset=["code"]).reset_index(drop=True)

        ETF_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(ETF_LIST_FILE, compression=PARQUET_COMPRESSION)
        return df

    # ========================================================================
    # 核心业务 2：批量下载全量 ETF
    # ========================================================================
    def fetch_all_etf_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        freq: str = "d",
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        批量下载全量 ETF 日线，落盘为 Parquet。
        AKShare 单次调用即返回上市至今全量数据，无需分页。

        参数:
            start_date: 起始日期 YYYY-MM-DD，默认取 config (2015-01-01)
            end_date:   结束日期 YYYY-MM-DD，默认今天
            freq:       "d"（日线），60min 暂不支持
            resume:     断点续传，跳过已存在的文件

        返回:
            {"success": N, "fail": N, "skipped": N, "empty": N, "failed_codes": [...]}
        """
        if freq == "60min":
            print("[警告] AKShare 免费接口不支持 60 分钟线，仅下载日线。")
            return {"success": 0, "fail": 0, "skipped": 0, "empty": 0, "failed_codes": []}

        start_date = start_date or self._default_start
        end_date = end_date or datetime.today().strftime("%Y-%m-%d")

        df_list = self.fetch_all_etf_list()
        codes: List[str] = df_list["code"].tolist()

        storage_dir: Path = ETF_DAILY_DIR
        storage_dir.mkdir(parents=True, exist_ok=True)

        result: Dict[str, Any] = {
            "success": 0, "fail": 0, "skipped": 0, "empty": 0, "failed_codes": []
        }
        total: int = len(codes)
        # 东方财富反爬：单请求间隔 0.3~0.8s，批次间额外休息
        req_delay_min, req_delay_max = 0.5, 1.5
        batch_size = 50
        batch_sleep = 2.0
        conn_error_cooldown = 0  # 连续连接失败计数，用于动态减速

        for i, code in enumerate(codes):
            p = storage_dir / f"{code}.parquet"
            if resume and p.exists():
                result["skipped"] += 1
                if (i + 1) % 200 == 0:
                    print(f"[进度] {i + 1}/{total} (跳过 {result['skipped']})")
                continue

            try:
                df = None
                last_error = None
                for attempt in range(MAX_RETRIES):
                    try:
                        # 搜狐 K 线 API（稳定，不易限速）
                        # code: cn_{code}，如 cn_510050
                        code_short = code.split(
                            ".")[-1] if "." in code else code.replace("sh", "").replace("sz", "")
                        url = "https://q.stock.sohu.com/hisHq"
                        params = {
                            "code": f"cn_{code_short}",
                            "start": start_date.replace("-", ""),
                            "end": end_date.replace("-", ""),
                            "stat": "1",
                            "order": "D",
                            "period": "d",
                        }
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                        }
                        r = requests.get(url, params=params,
                                         headers=headers, timeout=30)
                        if not r.text or not r.text.strip():
                            raise RuntimeError("搜狐API返回空响应")

                        raw = r.json()
                        # 搜狐返回: [{"status": 0, "hq": [[date,open,close,change%,low?,high?,volume,amount,...],...]}]
                        if not isinstance(raw, list) or len(raw) == 0:
                            df = pd.DataFrame()
                        else:
                            rows = raw[0].get("hq") or []
                            if not rows:
                                df = pd.DataFrame()
                            else:
                                df = pd.DataFrame(rows)
                                # 搜狐列顺序: date, open, close, change, change%, low, high, volume, amount, turnover%, amplitude
                                # 只取前9列足够: 0=date, 1=open, 2=close, 5=low, 6=high, 7=volume, 8=amount
                                if df.shape[1] >= 9:
                                    df = df.iloc[:, [0, 1, 6, 5, 2, 7, 8]]
                                    df.columns = [
                                        "date", "open", "high", "low", "close", "volume", "amount"]
                                else:
                                    df = pd.DataFrame()
                        break
                    except Exception as e:
                        last_error = e
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))

                if df is None:
                    conn_error_cooldown += 1
                    extra = min(conn_error_cooldown * 5, 60)
                    time.sleep(extra)
                    raise RuntimeError(f"下载失败({MAX_RETRIES}次重试): {last_error}")

                conn_error_cooldown = max(0, conn_error_cooldown - 1)

                if df.empty:
                    raise ValueError(
                        f"数据源返回空: {code} "
                        f"(params: code=cn_{code_short}, "
                        f"start={start_date.replace('-', '')}, end={end_date.replace('-', '')})"
                    )

                # 类型转换（强制 pd.to_numeric，防止字符串参与后续计算）
                for col in ["open", "high", "low", "close", "volume", "amount"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

                # 基础整理
                df.dropna(subset=["open", "high", "low",
                          "close", "volume"], how="all", inplace=True)
                df.drop_duplicates(subset=["date"], keep="first", inplace=True)
                df.sort_values("date", inplace=True)
                df.reset_index(drop=True, inplace=True)

                # 落盘
                df.to_parquet(p, compression=PARQUET_COMPRESSION)
                result["success"] += 1

            except Exception as e:
                result["fail"] += 1
                result["failed_codes"].append(code)
                print(f"[警告] {code} 失败: {e}")
                print(f"[TRACEBACK] {traceback.format_exc()}")

            # 每次请求间随机延迟，含连接错误惩罚
            delay = random.uniform(
                req_delay_min, req_delay_max) + conn_error_cooldown * 0.5
            time.sleep(delay)

            if (i + 1) % batch_size == 0:
                print(
                    f"[进度] {i + 1}/{total} "
                    f"(成功 {result['success']}, 跳过 {result['skipped']}, 失败 {result['fail']})"
                )
                time.sleep(batch_sleep)
                conn_error_cooldown = max(0, conn_error_cooldown - 1)  # 批次后降温

        self._update_meta(start_date, end_date, freq)
        print(
            f"[完成] 成功 {result['success']}, "
            f"跳过 {result['skipped']}, 空数据 {result.get('empty', 0)}, 失败 {result['fail']}"
        )
        return result

    # ========================================================================
    # 核心业务 3：自动增量更新
    # ========================================================================
    def auto_update(self) -> Dict[str, Any]:
        """
        自动增量更新日线。
        检测每只 ETF 本地最后日期，仅拉取缺失部分，追加合并到已有 Parquet。
        """
        df_list = self.fetch_all_etf_list()
        codes: List[str] = df_list["code"].tolist()
        today: str = datetime.today().strftime("%Y-%m-%d")

        result: Dict[str, Any] = {"updated": 0,
                                  "up_to_date": 0, "error": 0, "errors": []}
        batch_size = BAO_BATCH_SIZE * 2
        batch_sleep = BAO_BATCH_SLEEP / 2

        for i, code in enumerate(codes):
            p: Path = ETF_DAILY_DIR / f"{code}.parquet"
            try:
                if p.exists():
                    existing = pd.read_parquet(p)
                    if existing.empty:
                        last_date = "2015-01-01"
                    else:
                        last_date = existing["date"].max().strftime("%Y-%m-%d")

                    if last_date >= today:
                        result["up_to_date"] += 1
                        continue

                    next_day = (
                        pd.to_datetime(last_date) + pd.Timedelta(days=1)
                    ).strftime("%Y-%m-%d")

                    # 增量拉取（搜狐 API），含重试机制
                    code_short = code.split(
                        ".")[-1] if "." in code else code.replace("sh", "").replace("sz", "")
                    params = {"code": f"cn_{code_short}", "start": next_day.replace("-", ""),
                              "end": today.replace("-", ""), "stat": "1", "order": "D", "period": "d"}
                    df_new = pd.DataFrame()
                    last_error = None
                    for attempt in range(MAX_RETRIES):
                        try:
                            r = requests.get(
                                "https://q.stock.sohu.com/hisHq",
                                params=params,
                                headers={"User-Agent": "Mozilla/5.0"}, timeout=30,
                            )
                            if not r.text or not r.text.strip():
                                raise RuntimeError("搜狐API返回空响应")
                            raw = r.json()
                            if isinstance(raw, list) and len(raw) > 0:
                                rows = raw[0].get("hq") or []
                                if rows:
                                    df_new = pd.DataFrame(rows)
                                    if df_new.shape[1] >= 9:
                                        df_new = df_new.iloc[:, [
                                            0, 1, 6, 5, 2, 7, 8]]
                                        df_new.columns = [
                                            "date", "open", "high", "low", "close", "volume", "amount"]
                            break
                        except Exception as e:
                            last_error = e
                            if attempt < MAX_RETRIES - 1:
                                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))

                    if df_new.empty:
                        raise ValueError(
                            f"增量更新返回空: {code} "
                            f"(params: {params})"
                            + (f" | 最后错误: {last_error}" if last_error else "")
                        )

                    for col in ["open", "high", "low", "close", "volume", "amount"]:
                        if col in df_new.columns:
                            df_new[col] = pd.to_numeric(
                                df_new[col], errors="coerce")
                    df_new["date"] = pd.to_datetime(
                        df_new["date"], errors="coerce")

                    combined = (
                        pd.concat([existing, df_new], ignore_index=True)
                        .drop_duplicates(subset=["date"])
                        .sort_values("date")
                        .reset_index(drop=True)
                    )
                    combined.to_parquet(p, compression=PARQUET_COMPRESSION)
                else:
                    # 无本地文件，全量拉取（搜狐 API），含重试机制
                    code_short = code.split(
                        ".")[-1] if "." in code else code.replace("sh", "").replace("sz", "")
                    params = {"code": f"cn_{code_short}", "start": "20150101",
                              "end": today.replace("-", ""), "stat": "1", "order": "D", "period": "d"}
                    df_new = pd.DataFrame()
                    last_error = None
                    for attempt in range(MAX_RETRIES):
                        try:
                            r = requests.get(
                                "https://q.stock.sohu.com/hisHq",
                                params=params,
                                headers={"User-Agent": "Mozilla/5.0"}, timeout=30,
                            )
                            if not r.text or not r.text.strip():
                                raise RuntimeError("搜狐API返回空响应")
                            raw = r.json()
                            if isinstance(raw, list) and len(raw) > 0:
                                rows = raw[0].get("hq") or []
                                if rows:
                                    df_new = pd.DataFrame(rows)
                                    if df_new.shape[1] >= 9:
                                        df_new = df_new.iloc[:, [
                                            0, 1, 6, 5, 2, 7, 8]]
                                        df_new.columns = [
                                            "date", "open", "high", "low", "close", "volume", "amount"]
                            break
                        except Exception as e:
                            last_error = e
                            if attempt < MAX_RETRIES - 1:
                                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))

                    if df_new.empty:
                        raise ValueError(
                            f"全量拉取返回空: {code} "
                            f"(params: {params})"
                            + (f" | 最后错误: {last_error}" if last_error else "")
                        )

                    if not df_new.empty:
                        for col in ["open", "high", "low", "close", "volume", "amount"]:
                            if col in df_new.columns:
                                df_new[col] = pd.to_numeric(
                                    df_new[col], errors="coerce")
                        df_new["date"] = pd.to_datetime(
                            df_new["date"], errors="coerce")
                        p.parent.mkdir(parents=True, exist_ok=True)
                        df_new.to_parquet(p, compression=PARQUET_COMPRESSION)

                result["updated"] += 1
            except Exception as e:
                result["error"] += 1
                result["errors"].append({
                    "code": code,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

            # 每次请求间随机延迟（防封 IP）
            time.sleep(random.uniform(0.5, 1.5))

            if (i + 1) % batch_size == 0:
                print(f"[更新进度] {i + 1}/{len(codes)}")
                time.sleep(batch_sleep)

        self._update_meta("auto", today, "d")
        print(
            f"[更新完成] 更新 {result['updated']}, "
            f"已最新 {result['up_to_date']}, 错误 {result['error']}"
        )
        return result

    # ========================================================================
    # 内部方法：急速读取
    # ========================================================================
    def _fast_read(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        freq: str = "d",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        cache_key = f"{symbol}|{freq}"
        if cache_key in self._df_cache:
            df_cached = self._df_cache[cache_key]
            mask = (df_cached["date"] >= start_date) & (
                df_cached["date"] <= end_date)
            return df_cached.loc[mask] if columns is None else df_cached.loc[mask, columns]

        storage_dir = ETF_DAILY_DIR if freq == "d" else ETF_60MIN_DIR
        file_path = storage_dir / f"{symbol}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(
                f"本地无 {symbol} 的 {freq} 数据。请先调用 fetch_all_etf_data() 批量下载。"
            )

        df = pd.read_parquet(file_path)
        if df.empty:
            return df

        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if len(self._df_cache) > 50:
            self._df_cache.clear()
        self._df_cache[cache_key] = df

        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        return df.loc[mask] if columns is None else df.loc[mask, columns]

    # ========================================================================
    # 内部辅助
    # ========================================================================
    def _ensure_dirs(self) -> None:
        DATA_STORAGE.mkdir(parents=True, exist_ok=True)
        ETF_DAILY_DIR.mkdir(parents=True, exist_ok=True)
        ETF_60MIN_DIR.mkdir(parents=True, exist_ok=True)

    def _update_meta(self, start_date: str, end_date: str, freq: str) -> None:
        ETF_META_FILE.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "last_update": datetime.now(timezone.utc).isoformat(),
            "start_date": start_date,
            "end_date": end_date,
            "frequency": freq,
            "source": "sohu",
            "price_type": "raw_close",
            "survivorship_bias": True,
            "survivorship_note": "仅包含当前存续ETF，已退市/清盘ETF不在列表中",
        }
        with open(ETF_META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


# ============================================================================
# 便捷工厂函数
# ============================================================================

def create_etf_fetcher(config_path: Optional[str] = None) -> ETFDataFeed:
    return ETFDataFeed(config_path)


# ============================================================================
# 直接运行入口
# ============================================================================

if __name__ == "__main__":
    import sys

    fetcher = ETFDataFeed()

    # ---- 解析命令行参数 ----
    args = sys.argv[1:]
    force = "--force" in args
    args = [a for a in args if a != "--force"]

    freq = "d"
    start = None
    for a in args:
        if a in ("d", "60min"):
            freq = a
        elif len(a) == 10 and a[4] == "-":  # 日期 如 2015-01-01
            start = a

    if freq not in ("d", "60min"):
        print(
            f"用法: python {Path(__file__).name} [d|60min] [YYYY-MM-DD] [--force]")
        print(f"  d          → 日线（默认）")
        print(f"  60min      → 60 分钟线（暂不支持）")
        print(f"  2015-01-01 → 指定起始日期（默认 2015-01-01）")
        print(f"  --force    → 强制重新下载，覆盖已有文件")
        sys.exit(1)

    if freq == "60min":
        print("暂不支持 60 分钟线，切换为日线。")

    label = "日线"

    print(f"=== 1. 获取 ETF 代码列表 ===")
    etf_list = fetcher.fetch_all_etf_list()
    print(f"共 {len(etf_list)} 只 ETF")

    print(f"\n=== 2. 批量下载{label}数据 ===")
    if start:
        print(f"起始日期: {start}")
    if force:
        print("模式: 强制覆盖已有文件")
    result = fetcher.fetch_all_etf_data(
        freq="d", start_date=start, resume=not force)
    print(f"结果: {result}")

    print("\n=== 完成 ===")
