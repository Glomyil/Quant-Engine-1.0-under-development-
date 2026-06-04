import yaml
from pathlib import Path 
def load_config(config_path:str='config/settings.yaml')-> dict:

    path=Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path,'r',encoding='utf-8')as f:
        config=yaml.safe_load(f)
    return config
