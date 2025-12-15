from typing import Optional
from pathlib import Path

from veagentbench.evals.deepeval.test_run.cache import (
    TestRunCacheManager,
)
from dataclasses import dataclass

@dataclass
class ContextualCacheConfig:
    """
    增强的缓存配置，支持task和dataset上下文信息
    用于在多任务、多数据集场景下实现独立的缓存文件管理
    """
    task_name: Optional[str] = None
    dataset_name: Optional[str] = None
    agent_name: Optional[str] = None
    cache_base_dir: Optional[str] = None
    write_cache: bool = True
    use_cache: bool = False
    
    def get_cache_identifier(self) -> str:
        """
        生成基于上下文的缓存标识符
        用于区分不同task/dataset组合的缓存文件
        """
        parts = []
        if self.task_name:
            parts.append(f"task_{self.task_name}")
        if self.dataset_name:
            parts.append(f"dataset_{self.dataset_name}")
        if self.agent_name:
            parts.append(f"agent_{self.agent_name}")
        
        return "_".join(parts) if parts else "default"
    
    def get_cache_file_path(self, is_temp: bool = False) -> str:
        """
        获取缓存文件路径
        """
        from veagentbench.evals.deepeval.constants import HIDDEN_DIR
        from pathlib import Path
        
        base_name = f".veagenteval-cache-{self.get_cache_identifier()}.json"
        if is_temp:
            base_name = f".temp-veagenteval-cache-{self.get_cache_identifier()}.json"
            
        if self.cache_base_dir:
            # 如果指定了缓存基础目录，在该目录下创建子目录
            cache_dir = Path(self.cache_base_dir) / self.get_cache_identifier()
            cache_dir.mkdir(parents=True, exist_ok=True)
            return str(cache_dir / base_name)
        else:
            # 默认使用隐藏目录
            return f"{HIDDEN_DIR}/{base_name}"
    


class ContextualTestRunCacheManager(TestRunCacheManager):
    """
    增强的缓存管理器，支持task和dataset上下文
    为不同的task/dataset组合提供独立的缓存文件
    """
    
    def __init__(self, cache_config: Optional[ContextualCacheConfig] = None):
        super().__init__()
        self.cache_config = cache_config
        self._update_cache_paths()
    
    def _update_cache_paths(self):
        """根据缓存配置更新缓存文件路径"""
        if self.cache_config and hasattr(self.cache_config, 'get_cache_file_path'):
            self.cache_file_name = self.cache_config.get_cache_file_path(is_temp=False)
            self.temp_cache_file_name = self.cache_config.get_cache_file_path(is_temp=True)
            
            # 确保缓存目录存在
            cache_dir = Path(self.cache_file_name).parent
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    def set_cache_config(self, cache_config: ContextualCacheConfig):
        """设置缓存配置并更新路径"""
        self.cache_config = cache_config
        self._update_cache_paths()
    
    def get_cache_identifier(self) -> str:
        """获取当前缓存标识符"""
        if self.cache_config:
            return self.cache_config.get_cache_identifier()
        return "default"


# 全局上下文缓存管理器实例
global_contextual_test_run_cache_manager = ContextualTestRunCacheManager()


def get_contextual_cache_manager(cache_config: Optional[ContextualCacheConfig] = None) -> ContextualTestRunCacheManager:
    """
    获取上下文感知的缓存管理器
    如果提供了缓存配置，会创建或更新相应的缓存管理器
    """
    global global_contextual_test_run_cache_manager
    if cache_config:
        global_contextual_test_run_cache_manager.set_cache_config(cache_config)
   