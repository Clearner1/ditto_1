import re
import mysql.connector
from openai import OpenAI
from datetime import datetime
import hashlib
import time

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-2d50kWpmx7zcZyzXUcB94TXXBnxNZbGHx95zTqcCPHG7Luy8", 
    base_url="https://www.DMXapi.com/v1"
)

# MySQL配置
DB_CONFIG = {
    'host': '139.155.108.161',
    'user': 'ditto',
    'password': 'Ditto@123456',
    'database': 'features'
}

def get_hash(attribute_name, attribute_value):
    """生成属性名和属性值的组合哈希"""
    combined = f"{attribute_name}:{attribute_value}"
    return hashlib.sha256(combined.encode()).hexdigest()

def get_cached_feature(attribute_name, attribute_value):
    """从数据库中获取缓存的特征词"""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    
    hash_value = get_hash(attribute_name, attribute_value)
    cursor.execute('''
        SELECT feature_word FROM feature_cache 
        WHERE hash_value = %s
    ''', (hash_value,))
    
    result = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    return result['feature_word'] if result else None

def save_feature(attribute_name, attribute_value, feature_word):
    """保存特征词到数据库"""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    hash_value = get_hash(attribute_name, attribute_value)
    created_at = datetime.now()
    
    try:
        cursor.execute('''
            INSERT INTO feature_cache 
            (attribute_name, attribute_value, feature_word, hash_value, created_at)
            VALUES (%s, %s, %s, %s, %s)
        ''', (attribute_name, attribute_value, feature_word, hash_value, created_at))
        
        conn.commit()
    except mysql.connector.IntegrityError:
        # 如果哈希值已存在，忽略错误
        pass
    finally:
        cursor.close()
        conn.close()

def parse_entity(entity_str):
    """解析单个实体的属性（保持原有逻辑）"""
    entity_str = entity_str + ' COL'
    attributes = [f"COL {attr_str}" for attr_str 
                 in re.findall(r"(?<=COL ).*?(?= COL)", entity_str)]
    
    attr_dict = {}
    for attr in attributes:
        parts = attr.split(' VAL ')
        if len(parts) == 2:
            key = parts[0].replace('COL ', '').strip()
            value = parts[1].strip()
            attr_dict[key] = value
            
    return attr_dict

def generate_llm_prompt(attr_name, attr_value):
    """生成LLM提示词"""
    prompt = f"""作为资深啤酒品鉴专家，从啤酒属性中提取最具代表性的特征词。

要求:
1. 分析 {attr_name}: "{attr_value}"，提取1-2个最能体现啤酒特色的关键词
2. 优先选择体现啤酒风味、口感、酿造工艺的专业词汇
3. 可以基于你的啤酒专业知识，提取原文中未出现但更准确的特征词
4. 如果找不到明显特征，则返回基础风格(如Ale、Lager等)

示例：
输入: "Eruption Imperial Red Ale"
输出: "Strong, Malty" # 体现Imperial特点

输入: "Summer Light Lager" 
输出: "Crisp" # 体现Light Lager特点

仅返回特征词，无需解释。

请分析并提取:"""
    
    return prompt

def get_llm_response(prompt):
    """调用DeepSeek API获取响应"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a beer expert who extracts key features from beer names."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API调用错误: {str(e)}")
        return None

def clean_feature_word(feature_word):
    """清理特征词，去除双引号和多余的空格，保留1-2个词"""
    if feature_word is None:
        return None
    # 去除双引号和首尾空格
    cleaned = feature_word.strip('" \'')
    # 分割成词列表
    words = cleaned.split()
    # 只保留前两个词（如果有的话）
    if len(words) > 2:
        words = words[:2]
    # 用空格连接词
    cleaned = ' '.join(words)
    return cleaned

def extract_and_cache_feature(attribute_name, attribute_value):
    """提取特征并缓存"""
    # 首先尝试从缓存获取
    cached_feature = get_cached_feature(attribute_name, attribute_value)
    if cached_feature:
        return clean_feature_word(cached_feature), True  # True表示是从缓存获取的
    
    # 如果缓存中没有，调用LLM
    prompt = generate_llm_prompt(attribute_name, attribute_value)
    feature = get_llm_response(prompt)
    
    if feature:
        # 清理特征词
        cleaned_feature = clean_feature_word(feature)
        # 保存到数据库
        save_feature(attribute_name, attribute_value, cleaned_feature)
        time.sleep(3)  # 增加到3秒的延时
    
    return cleaned_feature, False  # False表示是新生成的

def process_line(line):
    """处理一行数据，提取两个实体的属性"""
    # 分割实体对
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None
    
    entity1 = parts[0]
    entity2 = parts[1]
    
    # 解析两个实体
    entity1_attrs = parse_entity(entity1)
    entity2_attrs = parse_entity(entity2)
    
    return entity1_attrs, entity2_attrs

def extract_features(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            try:
                entity1_attrs, entity2_attrs = process_line(line)
                
                print(f"\n=== 实体对 {i+1} ===")
                
                # 处理实体1
                print("实体1:")
                for attr_name in ['Beer_Name', 'Style']:
                    if attr_name in entity1_attrs:
                        attr_value = entity1_attrs[attr_name]
                        feature, is_cached = extract_and_cache_feature(attr_name, attr_value)
                        print(f"{attr_name}: {attr_value}")
                        print(f"提取的特征词: {feature} {'(已缓存)' if is_cached else '(新生成)'}")
                print()
                
                # 处理实体2
                print("实体2:")
                for attr_name in ['Beer_Name', 'Style']:
                    if attr_name in entity2_attrs:
                        attr_value = entity2_attrs[attr_name]
                        feature, is_cached = extract_and_cache_feature(attr_name, attr_value)
                        print(f"{attr_name}: {attr_value}")
                        print(f"提取的特征词: {feature} {'(已缓存)' if is_cached else '(新生成)'}")
                print()
                
                print("-" * 50)
                
            except Exception as e:
                print(f"Error processing line {i+1}: {str(e)}")

if __name__ == "__main__":
    # 切换到train.txt
    extract_features('valid.txt')