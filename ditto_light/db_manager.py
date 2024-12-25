from typing import Dict, Optional
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import colorama
from colorama import Fore, Style
from urllib.parse import quote_plus

# 初始化colorama
colorama.init()

# 配置logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理类"""
    
    def __init__(self, config: Dict):
        """初始化数据库管理器
        
        Args:
            config: 数据库配置，格式如：
                {
                    "host": "139.155.108.161",
                    "user": "ditto",
                    "password": "Ditto@123456",
                    "database": "number_perception"
                }
        """
        self.config = config
        self.engine = None
        self._init_engine()
    
    def _create_url(self, with_db: bool = True) -> str:
        """创建数据库URL
        
        Args:
            with_db: 是否包含数据库名
            
        Returns:
            数据库URL
        """
        # 使用 URL 编码处理特殊字符
        password = quote_plus(self.config['password'])
        
        base_url = f"mysql+pymysql://{self.config['user']}:{password}@{self.config['host']}"
        if with_db:
            return f"{base_url}/{self.config['database']}"
        return base_url
    
    def _init_engine(self):
        """初始化数据库引擎"""
        try:
            # 首先确保数据库存在
            self._ensure_database_exists()
            
            # 创建带连接池的引擎
            self.engine = create_engine(
                self._create_url(),
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600,
                echo=False
            )
            
            logger.info(f"{Fore.GREEN}数据库引擎初始化成功{Style.RESET_ALL}")
            
        except Exception as e:
            logger.error(f"{Fore.RED}数据库引擎初始化失败: {str(e)}{Style.RESET_ALL}")
            raise
    
    def _ensure_database_exists(self):
        """确保数据库存在"""
        # 创建一个临时引擎，不指定数据库
        temp_engine = create_engine(self._create_url(with_db=False))
        
        try:
            with temp_engine.connect() as conn:
                # 检查数据库是否存在
                result = conn.execute(text(
                    f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{self.config['database']}'"
                ))
                
                if not result.fetchone():
                    logger.info(f"{Fore.YELLOW}数据库不存在，正在创建...{Style.RESET_ALL}")
                    conn.execute(text(f"CREATE DATABASE {self.config['database']}"))
                    conn.commit()
                    logger.info(f"{Fore.GREEN}数据库创建成功{Style.RESET_ALL}")
                else:
                    logger.info(f"{Fore.CYAN}数据库已存在{Style.RESET_ALL}")
                    
        except SQLAlchemyError as e:
            logger.error(f"{Fore.RED}数据库检查/创建失败: {str(e)}{Style.RESET_ALL}")
            raise
        finally:
            temp_engine.dispose()
    
    def _ensure_table_exists(self):
        """确保必要的表存在"""
        try:
            with self.engine.connect() as conn:
                # 创建实体解析结果表
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS entity_segments (
                        entity_hash VARCHAR(64) PRIMARY KEY,
                        entity_text TEXT,
                        segments JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.commit()
                logger.info(f"{Fore.GREEN}数据表检查/创建成功{Style.RESET_ALL}")
                
        except SQLAlchemyError as e:
            logger.error(f"{Fore.RED}数据表检查/创建失败: {str(e)}{Style.RESET_ALL}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """执行SQL查询
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                row = result.fetchone()
                if row is None:
                    return None
                    
                # 处理SHOW TABLES查询
                if query.strip().upper().startswith("SHOW TABLES"):
                    return {"table": row[0]} if row else None
                    
                # 将结果转换为字典
                return dict(zip(result.keys(), row))
                
        except SQLAlchemyError as e:
            logger.error(f"{Fore.RED}查询执行失败: {str(e)}{Style.RESET_ALL}")
            return None
    
    def execute_write(self, query: str, params: Optional[Dict] = None) -> bool:
        """执行写入操作
        
        Args:
            query: SQL写入语句
            params: 写入参数
            
        Returns:
            是否成功
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(text(query), params or {})
                logger.info(f"{Fore.GREEN}SQL执行成功{Style.RESET_ALL}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"{Fore.RED}写入执行失败: {str(e)}{Style.RESET_ALL}")
            return False
    
    def dispose(self):
        """释放数据库连接池"""
        try:
            if hasattr(self, 'engine') and self.engine is not None:
                self.engine.dispose()
                logger.info(f"{Fore.CYAN}数据库连接池已释放{Style.RESET_ALL}")
        except Exception:
            # 忽略清理过程中的错误
            pass