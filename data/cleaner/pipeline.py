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

# 环节清单：顺序 = 执行顺序；元素 = (环节名, 模块)
PIPELINE = [
    # ("align", align),                   # ← align 写完就取消注释这一行
]

_START, _END = "2020-01-01", "2020-12-31"


def run_one(df, ctx, steps=None):
    """让一只股票的 df 依次走过清单里的每个环节 → (df, 每步报告)"""
    # ctx 契约：环节会读这两个键，缺了当场报错（别等到环节里 KeyError）
    for key in ("asset_type", "code"):
        if key not in ctx:
            raise KeyError(f"ctx 缺必填键 '{key}'（现有: {list(ctx)}）")

    reports = []
    for name, mod in (PIPELINE if steps is None else steps):
        before_cols = list(df.columns)          # 记账：进环节前的列（快照）
        rows_before = len(df)                   # 记账：进环节前的行数

        out = mod.run(df, {}, ctx)              # ★ 叫号：把 df 递给环节
        if out is None:
            raise TypeError(f"环节 '{name}' 返回了 None —— 忘了 return df？")
        df = out

        rep = base.new_step_report(name)        # 报告卡（base 预置了契约字段）
        rep["metrics"] = {"rows_before": rows_before, "rows_after": len(df)}
        rep["columns_added"] = [c for c in df.columns if c not in before_cols]
        extra = base.check_new_columns(before_cols, list(df.columns))
        if extra:
            base.add_warning(rep, f"出现未登记列（不在白名单）: {extra}")
        reports.append(rep)
    return df, reports


def run_pipeline(codes, start, end, steps=None):
    """逐只股票跑完整条线"""
    for code, df in loader.iter_frames(codes, start, end):
        ctx = {"universe": "stock", "asset_type": "stock", "code": code}
        df, reports = run_one(df, ctx, steps)

        desc = " | ".join(f"{r['name']} {r['metrics']['rows_before']}"
                          f"→{r['metrics']['rows_after']} +{len(r['columns_added'])}列"
                          for r in reports) or "（无环节）"
        print(f"  {code}: {desc}")


if __name__ == "__main__":
    codes = loader.list_codes()[:3]
    print("候选清单前 3 只:", codes)
    run_pipeline(codes, _START, _END)
    print("跑完 ✅")
