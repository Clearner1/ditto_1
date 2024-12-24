import mysql.connector
import re
from mysql.connector import Error
import sys

# MySQL配置
DB_CONFIG = {
    'host': '139.155.108.161',
    'user': 'ditto',
    'password': 'Ditto@123456',
    'database': 'features'
}

def get_feature_from_db(attribute_name, attribute_value):
    """从数据库中获取特征词"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # 打印查询信息
        print(f"\n正在查询: attribute_name='{attribute_name}', attribute_value='{attribute_value}'")
        
        cursor.execute('''
            SELECT feature_word FROM Beer_cache 
            WHERE attribute_name = %s AND attribute_value = %s
        ''', (attribute_name, attribute_value))
        
        result = cursor.fetchone()
        if result:
            print(f"找到特征词: {result['feature_word']}")
        else:
            print(f"未找到特征词")
        return result['feature_word'] if result else None
    
    except Error as e:
        print(f"数据库错误: {e}")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def extract_attribute_value(entity_text, attribute):
    """从实体文本中提取指定属性的值"""
    pattern = f"COL {attribute} VAL ([^\\t]*?)(?=\\s+COL|\\t|$)"
    match = re.search(pattern, entity_text)
    if match:
        value = match.group(1).strip()
        # 如果值中已经包含特征标记，去除它
        value = re.sub(r'\[.*?\]', '', value).strip()
        print(f"提取到 {attribute}: {value}")
        return value
    print(f"未提取到 {attribute}")
    return None

def replace_attribute_value(entity_text, attribute, old_value, new_value):
    """精确替换属性值"""
    try:
        # 构建一个更安全的模式
        escaped_old_value = re.escape(old_value)
        pattern = f"(COL {attribute} VAL ){escaped_old_value}"
        
        # 处理替换值中的特殊字符
        new_value = new_value.replace('\\', '\\\\')  # 处理反斜杠
        new_value = new_value.replace('$', '\\$')    # 处理美元符号
        
        result = re.sub(pattern, f"\\1{new_value}", entity_text)
        
        if result != entity_text:
            print(f"替换成功: {attribute}")
        else:
            print(f"替换失败: {attribute}")
        return result
    except Exception as e:
        print(f"替换时出错: {str(e)}")
        # 如果替换失败，返回原文本
        return entity_text

def process_entity(entity_text):
    """处理单个实体文本"""
    processed_text = entity_text
    
    # 处理 Beer_Name
    beer_name = extract_attribute_value(entity_text, "Beer_Name")
    if beer_name:
        feature = get_feature_from_db('Beer_Name', beer_name)
        if feature:
            processed_text = replace_attribute_value(
                processed_text, "Beer_Name", beer_name, f"{beer_name}[{feature}]"
            )
    
    # 处理 Style
    style = extract_attribute_value(processed_text, "Style")
    if style:
        feature = get_feature_from_db('Style', style)
        if feature:
            processed_text = replace_attribute_value(
                processed_text, "Style", style, f"{style}[{feature}]"
            )
    
    return processed_text

def process_line_with_features(line):
    """处理包含两个实体的行"""
    # 分割所有部分
    parts = line.split('\t')
    # 只取前两个部分（实体），忽略最后的匹配结果
    entities = parts[:2]
    # 保存匹配结果（如果存在）
    match_result = parts[2] if len(parts) > 2 else ""
    
    processed_entities = []
    
    print("\n处理新行...")
    for i, entity in enumerate(entities, 1):
        if entity.strip():  # 忽略空实体
            print(f"\n处理第{i}个实体:")
            processed_entity = process_entity(entity.strip())
            processed_entities.append(processed_entity)
        else:
            processed_entities.append(entity)
    
    # 添加回匹配结果
    if match_result:
        processed_entities.append(match_result)
    
    # 使用制表符重新连接
    return '\t'.join(processed_entities)

def process_file(input_file, output_file):
    """处理整个文件"""
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            total_lines = sum(1 for _ in infile)
        
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            processed_count = 0
            feature_added_count = 0
            
            for i, line in enumerate(infile, 1):
                original_line = line.strip()
                processed_line = process_line_with_features(original_line)
                
                if processed_line != original_line:
                    feature_added_count += 1
                
                outfile.write(processed_line + '\n')
                processed_count += 1
                
                # 打印进度
                if i % 100 == 0 or i == total_lines:
                    print(f"\n处理进度: {i}/{total_lines} ({(i/total_lines*100):.1f}%)")
            
            print(f"\n处理完成!")
            print(f"总行数: {processed_count}")
            print(f"添加特征的行数: {feature_added_count}")
                
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 定义要处理的数据集
    datasets = [
        ('train.txt', 'train_with_features.txt'),
        ('test.txt', 'test_with_features.txt'),
        ('valid.txt', 'valid_with_features.txt')
    ]
    
    # 处理每个数据集
    for input_file, output_file in datasets:
        print(f"\n开始处理数据集: {input_file}")
        print("=" * 50)
        try:
            process_file(input_file, output_file)
            print(f"数据集 {input_file} 处理完成，结果保存在 {output_file}")
            print("=" * 50)
        except Exception as e:
            print(f"处理数据集 {input_file} 时出错: {e}")
            print("继续处理下一个数据集...")
            continue