"""
数据清洗模块。

Pipeline 执行顺序：
    ⓪ ETFClassifier      — 产品分类筛选（Pipeline 级预筛选，非 Cleaner）
                            按资产类别（权益/债券/QDII/商品/货币）分类，
                            仅 A 股权益型 ETF 进入后续清洗流程。
    ① TimeAligner         — 时区强制 + 排序
    ② SuspensionChecker   — 停牌检测 + ffill（整行填充，禁止逐列）
    ③ AdjustChecker       — 除权日标记（输出 is_ex_dividend 列）
    ④ PriceValidator      — OHLC 校验 + 涨跌幅（跳过除权日）
    ⑤ SurvivorshipReporter — 幸存者偏差报告
    ⑥ DerivedFields       — 衍生列计算（close_unadjusted，非 adj_close）
"""
