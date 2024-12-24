import re

def parse_entity(entity_str):
    """解析单个实体的属性"""
    # 在字符串末尾添加COL以便匹配最后一个属性
    entity_str = entity_str + ' COL'
    
    # 使用正向查找提取属性字符串
    attributes = [f"COL {attr_str}" for attr_str 
                 in re.findall(r"(?<=COL ).*?(?= COL)", entity_str)]
    
    # 解析每个属性为键值对
    attr_dict = {}
    for attr in attributes:
        parts = attr.split(' VAL ')
        if len(parts) == 2:
            key = parts[0].replace('COL ', '').strip()
            value = parts[1].strip()
            attr_dict[key] = value
            
    return attr_dict

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

def analyze_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            try:
                entity1_attrs, entity2_attrs = process_line(line)
                
                print(f"\n=== 实体对 {i+1} ===")
                print("实体1:")
                for key, value in entity1_attrs.items():
                    print(f"{key} -> {value}")
                
                print("\n实体2:")
                for key, value in entity2_attrs.items():
                    print(f"{key} -> {value}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing line {i+1}: {str(e)}")

if __name__ == "__main__":
    # 分析train.txt文件
    analyze_file('train.txt')