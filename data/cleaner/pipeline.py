"""pipeline.py —— 执行器：把 df 按清单顺序递给每个清洗环节，并收集报告

职责：读环节清单 → 逐只取 df（loader）→ 按顺序叫号 → 记录每步指标 → 汇总
不做：任何清洗逻辑；不排序（顺序 = 清单顺序）；不读 parquet（loader 的活）

跑法：
    python -m data.cleaner.pipeline      # 手动冒烟：前 3 只、2020 区间
扩展：
    写完一个环节，就在 PIPELINE 里加一行 —— 其它地方都不用动
"""
from data import loader
from data.cleaner import base
from data.cleaner import align            # 环节①（写完后在 PIPELINE 里取消注释）
from data.cleaner import suspension
# 环节清单：顺序 = 执行顺序；元素 = (环节名, 模块)
PIPELINE = [("align", align), ("suspension", suspension)
            # ("align", align),                   # 这里填充对应的模块名称，方便后续效用
            ]

_START, _END = "2020-01-01", "2020-12-31"


def run_one(df, ctx, steps=None):
    """让一只股票的 df 依次走过清单里的每个环节 → (df, 每步报告)"""
    # ctx 契约：环节会读这两个键，缺了当场报错（别等到环节里 KeyError）
    for key in ("asset_type", "code"):  # 用于标记对应数据的类型并检测对应的列
        if key not in ctx:
            # 若数据出现错误，或报错并反应对应缺失列
            raise KeyError(f"ctx 缺必填键 '{key}'（现有: {list(ctx)}）")

    reports = []
    for name, mod in (PIPELINE if steps is None else steps):
        before_cols = list(df.columns)          # 记账：进环节前的列（快照）
        rows_before = len(df)                   # 记账：进环节前的行数

        out = mod.run(df, {}, ctx)              # 叫号：把 df 递给环节
        if out is None:
            # run函数未返回处理结果自动报错
            raise TypeError(f"环节 '{name}' 返回了 None —— 忘了 return df？")
        df = out  # 处理后的dataframe代替out并产出

        rep = base.new_step_report(name)        # 报告卡（base 预置了契约字段）
        rep["metrics"] = {"rows_before": rows_before, "rows_after": len(df)}
        rep["columns_added"] = [c for c in df.columns if c not in before_cols]
        extra = base.check_new_columns(before_cols, list(df.columns))
        if extra:
            base.add_warning(rep, f"出现未登记列（不在白名单）: {extra}")
        reports.append(rep)
    return df, reports  # 终输出步骤报错，记录故障数据


def run_pipeline(codes, start, end, steps=None):  # 定义了一个可以遍历codes内部编号进行整体处理的函数
    """逐只股票跑完整条线"""
    for code, df in loader.iter_frames(codes, start, end):  # 由于loader的这个函数用yield控制进程所以会先一条数据跑完pipeline后再导入下一个dataframe
        ctx = {"universe": "stock", "asset_type": "stock",
               "code": code}  # 这里标注了股票对应的类型
        df, reports = run_one(df, ctx, steps)  # 现在开始处理一个股票对应的数据

        desc = " | ".join(f"{r['name']} {r['metrics']['rows_before']}"
                          # 用于展现数据清洗的进度，便于追溯
                          f"→{r['metrics']['rows_after']} +{len(r['columns_added'])}列"
                          for r in reports) or "（无环节）"
        print(f"  {code}: {desc}")  # 最终展现进度


if __name__ == "__main__":
    codes = loader.list_codes()[:3]
    print("候选清单前 3 只:", codes)
    run_pipeline(codes, _START, _END)
    print("跑完 ✅")
