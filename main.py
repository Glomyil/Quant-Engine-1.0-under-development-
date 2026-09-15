# -*- coding: utf-8 -*-
"""
程序入口（占位/自检）
====================
当前职责：验证配置中心可用，打印品种=股票(A股含退市)的当前下载参数。
真正的 交易主循环 待 execution 模块实现后接入（见 工作流.txt / QD_GUIDE.md）。

修正记录 2026-09-07:
  原 main.py 引用 config.loader（源文件不存在，仅残留旧 pyc，必报错）。
  改为统一走 config.config_manager（热加载单例），保持全仓库配置同源。
"""

from config.config_manager import config


def main() -> None:
    app_name: str = str(config.get("app_name", "Quant-1.0"))
    start: str = str(config.get("stock.fetch.start_date", "2015-01-01"))
    workers: int = int(config.get("stock.fetch.workers", 8))

    print(f"应用: {app_name}")
    print(f"品种: A股(含退市) | 日线起始 {start} | 下载并发 {workers} 进程")
    print("配置中心自检 OK — 交易主循环待 execution 模块接入")


if __name__ == "__main__":
    main()
