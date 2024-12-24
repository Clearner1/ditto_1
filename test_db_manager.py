from ditto_light.db_manager import DatabaseManager
import json

def test_database_manager():
    # 数据库配置
    db_config = {
        "host": "139.155.108.161",
        "user": "ditto",
        "password": "Ditto@123456",
        "database": "number_perception"
    }
    
    # 创建数据库管理器
    db_manager = DatabaseManager(db_config)
    
    # 确保表存在
    db_manager._ensure_table_exists()
    
    # 测试写入操作
    test_data = {
        'entity_hash': 'test_hash_123',
        'entity_text': 'Canon EOS 70D Digital Camera',
        'segments': json.dumps([
            {
                'string': '70D',
                'type': 'PRODUCT_ID'
            }
        ])
    }
    
    success = db_manager.execute_write("""
        INSERT INTO entity_segments (entity_hash, entity_text, segments)
        VALUES (:entity_hash, :entity_text, :segments)
        ON DUPLICATE KEY UPDATE
            segments = VALUES(segments),
            created_at = CURRENT_TIMESTAMP
    """, test_data)
    
    print(f"写入测试数据{'成功' if success else '失败'}")
    
    # 测试查询操作
    result = db_manager.execute_query("""
        SELECT * FROM entity_segments WHERE entity_hash = :entity_hash
    """, {'entity_hash': 'test_hash_123'})
    
    if result:
        print("\n查询结果:")
        print(f"实体文本: {result['entity_text']}")
        print(f"解析结果: {result['segments']}")
    
    # 释放连接池
    db_manager.dispose()

if __name__ == "__main__":
    test_database_manager()