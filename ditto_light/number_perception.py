from typing import List, Tuple, Dict
import json
from openai import OpenAI
from dataclasses import dataclass

@dataclass
class NumberSegment:
    string: str
    type: str

class NumberPerception:
    """数字感知匹配模块"""
    
    # 数字类型枚举
    TYPE_PURE_NUMBER = "PURE_NUMBER"  # 纯数字
    TYPE_PRODUCT_ID = "PRODUCT_ID"    # 产品型号
    TYPE_SPEC = "SPEC"                # 技术规格
    TYPE_VERSION = "VERSION"          # 版本号
    TYPE_QUANTITY = "QUANTITY"        # 数量单位
    
    def __init__(self, api_key: str):
        """初始化数字感知模块
        
        Args:
            api_key: DeepSeek API密钥
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://www.DMXapi.com/v1"
        )
        
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

    def parse(self, text: str) -> List[NumberSegment]:
        """解析文本中的数字片段
        
        Args:
            text: 输入文本
            
        Returns:
            数字片段列表，每个片段包含字符串和类型
        """
        # 构建完整提示
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_template.format(text=text)}
        ]
        
        # 调用 LLM
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            
            # 解析返回的JSON
            result = json.loads(response.choices[0].message.content)
            return [NumberSegment(**item) for item in result]
            
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return []

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> float:
        """计算两个字符串的归一化编辑距离
        
        Args:
            s1: 第一个字符串
            s2: 第二个字符串
            
        Returns:
            归一化的编辑距离，范围[0,1]，1表示完全相同
        """
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
        """计算两个实体的数字感知相似度
        
        Args:
            segments1: 第一个实体的数字片段列表
            segments2: 第二个实体的数字片段列表
            
        Returns:
            相似度分数，范围[0,1]
        """
        if not segments1 or not segments2:
            return 0.0
            
        total_sim = 0.0
        
        # 计算所有可能的片段对的相似度
        for seg1 in segments1:
            for seg2 in segments2:
                # 只有类型相同的片段才计算相似度
                if seg1.type == seg2.type:
                    sim = self.levenshtein_distance(seg1.string, seg2.string)
                    total_sim += sim
        
        # 归一化
        return total_sim / (len(segments1) * len(segments2))

    def compare_entities(self, entity1: str, entity2: str) -> float:
        """比较两个实体的数字感知相似度
        
        Args:
            entity1: 第一个实体文本
            entity2: 第二个实体文本
            
        Returns:
            相似度分数，范围[0,1]
        """
        # 解析两个实体
        segments1 = self.parse(entity1)
        segments2 = self.parse(entity2)
        
        # 计算相似度
        return self.calculate_similarity(segments1, segments2) 