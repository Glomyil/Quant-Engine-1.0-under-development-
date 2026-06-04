from config.loader import load_config
cfg = load_config()
print(f"准备交易 {cfg['data']['symbol']}，模式 {cfg['mode']}")