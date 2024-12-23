from number_perception import NumberPerception

def test_number_perception():
    # 初始化模块
    np = NumberPerception(api_key="sk-2d50kWpmx7zcZyzXUcB94TXXBnxNZbGHx95zTqcCPHG7Luy8")
    
    # 测试用例
    entity1 = "iPhone 13 Pro Max 256GB"
    entity2 = "iPhone 13 Pro 256GB"
    
    # 解析实体
    segments1 = np.parse(entity1)
    print("\nEntity 1 segments:")
    for seg in segments1:
        print(f"- {seg.string}: {seg.type}")
        
    segments2 = np.parse(entity2)
    print("\nEntity 2 segments:")
    for seg in segments2:
        print(f"- {seg.string}: {seg.type}")
    
    # 计算相似度
    similarity = np.compare_entities(entity1, entity2)
    print(f"\nSimilarity score: {similarity:.4f}")

if __name__ == "__main__":
    test_number_perception()