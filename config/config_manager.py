# 自动热加载的配置管理器
import os  # 导入os用以读取文件修改时间戳
import yaml  # 将yaml文件转换为python字典
import time
from pathlib import Path  # 用更好的方式跨平台处理文件


class HotConfigManager:  # 定义一个大类
    _instance = None  # 单例模式
    _config = {}  # 以字典形式存储yaml文件内容
    _yaml_path = None  # 存储yaml文件路径
    _last_mtime = 0.0  # 定义一个类变量

    def __new__(cls): #重新定义参数
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance  # 假如instance是none，则创建一个新的实例对象赋值给cls-instance
    
# 以此保证多少个pyhton文件里面加载的都是同一个实例对象，确保文件参数来源同源

    def _initialize(self):
        self._yaml_path = Path(__file__).parent / "application.yml"
        self._load_config()
        
# 读取该文件夹下的yaml文件，将配置数据加载进内存

    def _load_config(self):
        if not self._yaml_path.exists():
            self._config = {}
            self._last_mtime = 0.0
            return
        with open(self._yaml_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f) or {}
        self._last_mtime = os.path.getmtime(self._yaml_path)  # 纪录修改文件变量的时间戳
        
# 判断文件是否存在，不存在则将配置设为空，若存在则将yaml文件的比那辆用with open只读打开，并将其解析为python字典

    def _check_and_reload(self):
        if not self._yaml_path.exists():
            return
        current_mtime = os.path.getmtime(self._yaml_path)#获取当前文件时间戳
        if current_mtime != self._last_mtime:
            print(f"检测到 application.yml 已被修改，自动重新加载...")
            self._load_config()
            
#通过对比修改后的时间戳来判断变量是否有改变，有则重新读取yaml文件内的变量

    def get(self, key, default=None):#从字典中获取键和对应的值
        self._check_and_reload() ###这个时候调用热插拔模块，去看一眼文件是否有被修改
        
        keys = key.split('.')  #被储存为列表
        value = self._config
        for k in keys:  #读取全部
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default
    
#对文件进行遍历，如果有问题，则返回默认值，否则返回对应的参数的值

    def set(self, key, value):
        keys = key.split('.') #对键进行分割
        d = self._config
        for k in keys[:-1]:
            d = d.setdefault(k, {}) 
        d[keys[-1]] = value #返回那个键对应的值
        
#用于在内存中动态修改配置，并从键值对中获取对应的变量对应的参数

    def save(self):
        with open(self._yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)
        self._last_mtime = os.path.getmtime(self._yaml_path)
        
# 编码格式，语言等基本设置


# 模块级单例导出 —— 所有模块统一通过 `from config.config_manager import config` 获取同一个配置实例
config = HotConfigManager()
