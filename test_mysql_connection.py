import mysql.connector
from colorama import init, Fore, Style

# 初始化colorama
init()

def test_mysql_connection():
    # 数据库配置
    db_config = {
        'host': '139.155.108.161',  # 服务器IP
        'user': 'ditto',             # 用户名
        'password': 'Ditto@123456',       # 密码
        'port': 3306               # MySQL端口号
    }
    
    try:
        print(f"{Fore.CYAN}正在连接MySQL服务器...{Style.RESET_ALL}")
        
        # 尝试连接
        conn = mysql.connector.connect(**db_config)
        
        if conn.is_connected():
            print(f"{Fore.GREEN}连接成功!{Style.RESET_ALL}")
            
            # 获取服务器信息
            db_info = conn.get_server_info()
            print(f"{Fore.YELLOW}MySQL版本: {db_info}{Style.RESET_ALL}")
            
            # 创建游标
            cursor = conn.cursor()
            
            # 执行查询
            cursor.execute("SHOW DATABASES")
            
            # 获取所有数据库
            print(f"\n{Fore.CYAN}现有数据库列表:{Style.RESET_ALL}")
            for db in cursor:
                print(f"  - {db[0]}")
            
            # 尝试创建数据库
            try:
                print(f"\n{Fore.YELLOW}尝试创建数据库 'number_perception'...{Style.RESET_ALL}")
                cursor.execute("CREATE DATABASE IF NOT EXISTS number_perception")
                print(f"{Fore.GREEN}数据库创建成功或已存在{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}创建数据库失败: {str(e)}{Style.RESET_ALL}")
            
            # 关闭连接
            cursor.close()
            conn.close()
            print(f"\n{Fore.GREEN}连接已关闭{Style.RESET_ALL}")
            
    except Exception as e:
        print(f"{Fore.RED}连接失败: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    test_mysql_connection() 