from typing import List, Dict
import json
from openai import OpenAI
from dataclasses import dataclass
import logging
import colorama
from colorama import Fore, Style
import hashlib
import time
from .db_manager import DatabaseManager

# 初始化colorama
colorama.init()

# 配置logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NumberSegment:
    string: str
    type: str
    
    def __str__(self):
        color_map = {
            "PURE_NUMBER": Fore.CYAN,
            "PRODUCT_ID": Fore.GREEN,
            "SPEC": Fore.YELLOW,
            "VERSION": Fore.MAGENTA,
            "QUANTITY": Fore.BLUE
        }
        color = color_map.get(self.type, Fore.WHITE)
        return f"{color}[{self.string} ({self.type})]{Style.RESET_ALL}"
    
    def to_dict(self):
        return {
            "string": self.string,
            "type": self.type
        }

class NumberPerceptionCache:
    """带缓存的数字感知匹配模块"""
    
    # 数字类型枚举
    TYPE_PURE_NUMBER = "PURE_NUMBER"  # 纯数字
    TYPE_PRODUCT_ID = "PRODUCT_ID"    # 产品型号
    TYPE_SPEC = "SPEC"                # 技术规格
    TYPE_VERSION = "VERSION"          # 版本号
    TYPE_QUANTITY = "QUANTITY"        # 数量单位
    
    def __init__(self, api_key: str, db_config: Dict):
        """初始化数字感知模块
        
        Args:
            api_key: DeepSeek API密钥
            db_config: MySQL数据库配置，格式如：
                {
                    "host": "139.155.108.161",
                    "user": "ditto",
                    "password": "Ditto@123456",
                    "database": "number_perception"
                }
        """
        logger.info(f"{Fore.CYAN}初始化数字感知模块...{Style.RESET_ALL}")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://www.DMXapi.com/v1"
        )
        
        # 初始化数据库管理器
        logger.info(f"{Fore.CYAN}初始化数据库连接...{Style.RESET_ALL}")
        self.db_manager = DatabaseManager(db_config)
        
        # 确保数据表存在
        logger.info(f"{Fore.CYAN}检查数据表...{Style.RESET_ALL}")
        self.db_manager._ensure_table_exists()
        
        # 测试数据库连接和表是否正确创建
        logger.info(f"{Fore.CYAN}测试数据库连接...{Style.RESET_ALL}")
        try:
            test_result = self.db_manager.execute_query("SHOW TABLES LIKE 'entity_segments'")
            if test_result:
                logger.info(f"{Fore.GREEN}数据库连接测试成功，entity_segments表存在{Style.RESET_ALL}")
            else:
                logger.error(f"{Fore.RED}数据库连接测试失败，entity_segments表不存在{Style.RESET_ALL}")
                raise Exception("数据表创建失败")
            
            # 测试写入
            test_write = self.db_manager.execute_write("""
                INSERT INTO entity_segments (entity_hash, entity_text, segments)
                VALUES ('test_hash', 'test_text', '[]')
                ON DUPLICATE KEY UPDATE created_at = CURRENT_TIMESTAMP
            """)
            if test_write:
                logger.info(f"{Fore.GREEN}数据库写入测试成功{Style.RESET_ALL}")
            else:
                logger.error(f"{Fore.RED}数据库写入测试失败{Style.RESET_ALL}")
                raise Exception("数据库写入测试失败")
            
        except Exception as e:
            logger.error(f"{Fore.RED}数据库测试失败: {str(e)}{Style.RESET_ALL}")
            raise
        
        # 系统提示
        self.system_prompt = """你是一个专门识别和分类数字信息的AI助手。你的任务是分析文本中的数字相关信息，并以严格的JSON数组格式返回结果。
请记住：
1. 返回的必须是一个合法的JSON数组
2. 数组中的每个元素必须包含 "string" 和 "type" 两个字段
3. 不要返回任何其他解释性文字，只返回JSON数组"""
        
        # 用户提示模板
        self.user_prompt_template = """分析以下文本中的所有数字相关信息：
{text}

严格按照以下JSON数组格式返回结果（不要返回任何其他文字）：
[
  {{
    "string": "识别的字符串",
    "type": "数字类型"
  }}
]

数字类型必须是以下之一：
- PURE_NUMBER: 纯数字，如 "100", "2.5"
- PRODUCT_ID: 产品型号，如 "RTX3080", "iPhone 13"
- SPEC: 技术规格，如 "8GB", "2TB", "3.5mm"
- VERSION: 版本号，如 "v2.0", "Windows 11"
- QUANTITY: 数量单位，如 "双核", "三件套"

注意：
1. 必须返回合法的JSON数组
2. 不要添加任何解释或额外文字
3. 每个识别出的片段必须包含 string 和 type 两个字段"""

    def _get_entity_hash(self, text: str) -> str:
        """计算实体文本的哈希值"""
        return hashlib.sha256(text.encode()).hexdigest()

    def _get_cached_segments(self, text: str) -> List[NumberSegment]:
        """从缓存中获取实体的解析结果"""
        entity_hash = self._get_entity_hash(text)
        result = self.db_manager.execute_query(
            "SELECT segments FROM entity_segments WHERE entity_hash = :entity_hash",
            {"entity_hash": entity_hash}
        )
        
        if result:
            segments_data = json.loads(result['segments'])
            return [NumberSegment(**item) for item in segments_data]
        return None

    def _cache_segments(self, text: str, segments: List[NumberSegment]):
        """缓存实体的解析结果"""
        try:
            entity_hash = self._get_entity_hash(text)
            segments_json = json.dumps([seg.to_dict() for seg in segments])
            
            success = self.db_manager.execute_write("""
                INSERT INTO entity_segments (entity_hash, entity_text, segments)
                VALUES (:entity_hash, :entity_text, :segments)
                ON DUPLICATE KEY UPDATE
                    segments = VALUES(segments),
                    created_at = CURRENT_TIMESTAMP
            """, {
                "entity_hash": entity_hash,
                "entity_text": text,
                "segments": segments_json
            })
            
            if success:
                logger.info(f"{Fore.GREEN}成功缓存结果{Style.RESET_ALL}")
            else:
                logger.error(f"{Fore.RED}缓存结果失败{Style.RESET_ALL}")
            
        except Exception as e:
            logger.error(f"{Fore.RED}缓存过程发生错误: {e}{Style.RESET_ALL}")
            logger.error(f"错误详情: {str(e)}")

    def parse(self, text: str) -> List[NumberSegment]:
        """解析文本中的数字片段"""
        logger.info(f"\n{Fore.YELLOW}开始解析文本:{Style.RESET_ALL} {text}")
        
        # 尝试从缓存获取
        cached_segments = self._get_cached_segments(text)
        if cached_segments is not None:
            logger.info(f"{Fore.GREEN}从缓存获取解析结果:{Style.RESET_ALL}")
            for segment in cached_segments:
                logger.info(f"  {segment}")
            return cached_segments
        
        # 构建完整提示
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_template.format(text=text)}
        ]
        
        # 调用 LLM
        logger.info(f"{Fore.CYAN}正在调用LLM服务...{Style.RESET_ALL}")
        try:
            # 添加3秒延时
            time.sleep(3)
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
        except Exception as e:
            logger.error(f"{Fore.RED}LLM调用失败: {e}{Style.RESET_ALL}")
            return []

        # 解析结果
        try:
            result = json.loads(response.choices[0].message.content)
            segments = [NumberSegment(**item) for item in result]
            
            # 打印识别结果
            logger.info(f"{Fore.GREEN}识别结果:{Style.RESET_ALL}")
            for segment in segments:
                logger.info(f"  {segment}")
            
            # 尝试缓存结果
            logger.info(f"{Fore.CYAN}正在缓存解析结果...{Style.RESET_ALL}")
            self._cache_segments(text, segments)
            
            return segments
            
        except json.JSONDecodeError as e:
            logger.error(f"{Fore.RED}JSON解析失败: {e}{Style.RESET_ALL}")
            return []
        except Exception as e:
            logger.error(f"{Fore.RED}结果处理失败: {e}{Style.RESET_ALL}")
            return []

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> float:
        """计算两个字符串的归一化编辑距离"""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
            
        # 创建矩阵
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        # 动态规划填充
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # 删除
                    dp[i][j-1] + 1,      # 插入
                    dp[i-1][j-1] + cost  # 替换
                )
        
        # 归一化
        max_len = max(m, n)
        if max_len == 0:
            return 1.0
        return 1 - (dp[m][n] / max_len)

    def calculate_similarity(self, segments1: List[NumberSegment], segments2: List[NumberSegment]) -> float:
        """计算两个实体的数字感知相似度"""
        if not segments1 or not segments2:
            logger.info(f"{Fore.YELLOW}至少一个实体没有数字片段，相似度为0{Style.RESET_ALL}")
            return 0.0
            
        total_sim = 0.0
        matches = []
        
        # 计算所有可能的片段对的相似度
        for seg1 in segments1:
            for seg2 in segments2:
                # 只有类型相同的片段才计算相似度
                if seg1.type == seg2.type:
                    sim = self.levenshtein_distance(seg1.string, seg2.string)
                    total_sim += sim
                    if sim > 0:
                        matches.append((seg1, seg2, sim))
        
        # 打印匹配结果
        if matches:
            logger.info(f"\n{Fore.GREEN}匹配详情:{Style.RESET_ALL}")
            for seg1, seg2, sim in matches:
                logger.info(f"  {seg1} ←→ {seg2} = {sim:.3f}")
        
        # 归一化
        similarity = total_sim / (len(segments1) * len(segments2))
        logger.info(f"{Fore.CYAN}最终相似度: {similarity:.3f}{Style.RESET_ALL}")
        return similarity

    def compare_entities(self, entity1: str, entity2: str) -> float:
        """比较两个实体的数字感知相似度"""
        logger.info(f"\n{Fore.MAGENTA}开始比较实体:{Style.RESET_ALL}")
        logger.info(f"实体1: {entity1}")
        logger.info(f"实体2: {entity2}")
        
        # 解析两个实体（会自动使用缓存）
        logger.info(f"\n{Fore.YELLOW}解析实体1:{Style.RESET_ALL}")
        segments1 = self.parse(entity1)
        
        logger.info(f"\n{Fore.YELLOW}解析实体2:{Style.RESET_ALL}")
        segments2 = self.parse(entity2)
        
        # 计算相似度
        logger.info(f"\n{Fore.YELLOW}计算相似度:{Style.RESET_ALL}")
        return self.calculate_similarity(segments1, segments2)
        
    def __del__(self):
        """析构函数，确保释放数据库连接池"""
        try:
            if hasattr(self, 'db_manager') and self.db_manager is not None:
                self.db_manager.dispose()
        except Exception:
            # 忽略清理过程中的错误
            pass