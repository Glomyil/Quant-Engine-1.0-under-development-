"""loader.py —— 数据层：parquet → DataFrame（df 的第一个生产者）

职责：选文件 → read_parquet → 区间裁剪(S0) → 守卫 → 逐只交出。
不做：排序/去重/类型修正（align 的活）、复权/指标（adjust/derived 的活）。
不进 yml 的 pipeline 列表、不写 run(df, params, ctx) 签名。

用法：
    python -m data.loader          # 手动跑一只看效果（模块方式运行 → 不需要改 sys.path）
    from data.loader import list_codes, load_one, iter_frames
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from config.constants import STOCK_DAILY_DIR
from data.cleaner import base


# 此模块用于呈现当前存储数据的文件夹内部的数据文件清单，即股票代码对应的清单
def list_codes(store: Path = STOCK_DAILY_DIR) -> list[str]:
    "候选清单：目录里有数据的股票，含退市股不漏样本"
    return sorted(p.stem for p in store.glob("*.parquet"))  # 用列表存储方便后续调用


def load_one(code: str, start: str, end: str,
             store: Path = STOCK_DAILY_DIR) -> Optional[pd.DataFrame]:
    """读一只并裁剪到 [start, end]；结构错→raise，区间内无行→None"""
    path = store / f"{code}.parquet"  # 用于明确该文件所在的位置
    if not path.exists():
        return None                                   # 文件不存在 = 业务现象

    # 通过pandas将之前已经定义的对应文件所在路径path以及对应代码名称读入内存，方便读取
    df = pd.read_parquet(path)

    # 调用base.py内校验缺失列的函数，缺列 = 结构错误, 必须 raise
    base.check_required_columns(df)

    # date 列统一成 datetime 类型（原可能是字符串/整数，无法与 Timestamp 比较）
    # 注意：errors="coerce" 会把脏值变 NaT，后续比较时静默剔除
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 此时要对筛选的时间进行序列分析，因而利用pandas将start和end的部分转换为时间戳来进行对比
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)

    # 这里设定了条件，明确时间戳内对数据进行处理，只保留筛选出来的行
    df = df[(df["date"] >= t0) & (df["date"] <= t1)]

    if len(df) == 0:
        # 这里用于标注特殊情况 行空 = 业务现象（未上市/整段停牌）便于后续处理进行跳过
        return None

    # pandas实际上读取进去内存的数据对应的索引值在数据筛选后并不是完全不变的，所以需要对新data frame内的数据进行索引值的重置
    return df.reset_index(drop=True)


def iter_frames(codes: list[str], start: str, end: str,  # 此函数用于将对应parquet文件的编号和对应的data frame捆绑在一起
                store: Path = STOCK_DAILY_DIR) -> Iterator[tuple[str, pd.DataFrame]]:
    """逐只产出 (code, df)：生成器 → 内存里同时只有一只股票"""
    for code in codes:  # for 循环调用codes内部的编号，设置为code然后将对应的code在下一步进行处理

        # 此时调用前面的load函数将对应的股票以data frame的形式读入内存
        df = load_one(code, start, end, store)

        if df is None:  # 为了保证鲁棒性，空data frame用none标注
            continue
        yield code, df  # 在数据的处理的时候，return会一次性读入所有的股票，内存会撑不住，所以采用yield一只一只地读取，处理好上一个再去处理下一个


if __name__ == "__main__":
    # 手动冒烟：覆盖 ① 主路径 ② 边界 ③ 异常分支（行空→None）
    # ⚠️ 参数必须"能走到被测代码行"：用不存在的代码会提前 return，测不到区间裁剪
    codes = list_codes()
    print("候选清单:", len(codes))                                      # 期望 5471

    df = load_one("sh.600000", "2020-01-01",
                  "2020-12-31")               # ① 主路径
    # 期望 (243, 11)
    print("正常区间 shape:", None if df is None else df.shape)
    assert df is not None and len(df) > 0, "主路径应当返回数据"
    assert df["date"].min() >= pd.Timestamp(
        "2020-01-01")                # ② 边界（含首尾）
    assert df["date"].max() <= pd.Timestamp("2020-12-31")

    print("区间外:", load_one("sh.600000", "2030-01-01", "2030-12-31"))   # ③ 行空分支
    # sh.600005 于 2017 退市
    print("退市股:", load_one("sh.600005", "2020-01-01", "2020-12-31"))
    print("不存在:", load_one("sh.999999", "2020-01-01", "2020-12-31"))
    print("冒烟通过 ✅")
