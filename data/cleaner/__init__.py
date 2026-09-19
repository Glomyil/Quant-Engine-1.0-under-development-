"""
数据清洗模块（data.cleaner）。

角色分工（详情见 教学文档/数据契约.md，此处不复制）：
    base.py      工具箱 + 契约注册表（COLUMN_WHITELIST）——不碰数据
    pipeline.py  执行器：编排顺序、外部记账、生成报告
    其余文件      环节：统一签名 run(df, params, ctx) -> df，每个只做一件事

执行顺序（PIPELINE 里的顺序就是实际顺序）：
    ① align        整备：日期规范化 / 丢坏日期行 / 同日去重 / 升序 / 数值类型     [S1 · 已实现]
    ② suspension   停牌识别与打标：is_suspended / suspension_days               [S1 · 已实现]

    ③ price        价格校验：OHLC 关系 / 非法价格 / 跨列一致性 / VWAP 物理校验   [S1 · 未实现]
    ④ st           ST 段打标：is_st                                            [S1 · 未实现]
    ⑤ adjust       复权：hfq_close / is_ex_div / 因子单调性 / 停牌日 hfq 不变量   [S2 · 未实现]
    ⑥ derived      派生：收益率 / 波动率等                                      [S3 · 未实现]
    ⑦ liquidity    流动性打标：low_liquidity（默认禁用，阈值待回测）             [S1/S3 · 未实现]

    ⚠️ ③~⑦ 的最终顺序尚未定稿，两个已确定的约束：
       - derived 必须在 adjust 之后（收益率要用后复权价）
       - 涉及"涨跌幅"的校验（极端涨跌候选）需要 adjust 产出的除权日标记

三条铁律（违反任何一条都会造成"不报错的错"）：
    1. 只标记，不删行（唯一例外：align 丢掉日期无法解析的行）
    2. 不填充价格/成交量（禁止 ffill 停牌期、禁止 fillna(0)）
    3. 一列只有一个产出者；新增列必须先登记 base.COLUMN_WHITELIST

配套文档（内容不在这里重复）：
    教学文档/数据契约.md         —— 哪个文件负责哪一列（唯一权威）
    教学文档/数据清洗事项总表.md  —— 事项编号、S 阶段、优先级、落点、状态
    教学文档/项目状态卡.md        —— 当前进度
    ops/health_check.py          —— 全库体检（"最大限度干净"的判据）
"""
