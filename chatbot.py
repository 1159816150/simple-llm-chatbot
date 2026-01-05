import os
import requests
from dotenv import load_dotenv
from typing import List, Dict

def get_tongyi_headers() -> Dict[str, str]:
    """
    构建通义千问API请求头（加载环境变量，配置API密钥）
    """
    # 加载.env文件中的环境变量
    load_dotenv()
    
    # 获取API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未配置DASHSCOPE_API_KEY，请在.env文件中填写你的通义千问API密钥")
    
    # 构建并返回请求头
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

def init_chat_history() -> List[Dict[str, str]]:
    """
    初始化聊天历史（设置系统角色，定义机器人行为）
    """
    return [
        {
            "role": "system",
            "content": "你是一个友好、耐心的智能聊天助手，能够回答用户的各种问题，语言简洁易懂。"
        }
    ]

def chat_with_tongyi(headers: Dict[str, str], chat_history: List[Dict[str, str]], user_input: str) -> str:
    """
    调用通义千问API，获取机器人回复
    :param headers: 请求头（包含API密钥）
    :param chat_history: 聊天历史列表
    :param user_input: 用户当前输入
    :return: 机器人回复内容
    """
    # 获取模型名称（从环境变量加载，默认qwen-turbo）
    model = os.getenv("TONGYI_MODEL", "qwen-turbo")
    
    # 通义千问API端点
    api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    
    # 将用户输入添加到聊天历史
    chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # 构建请求体
    request_body = {
        "model": model,
        "input": {
            "messages": chat_history
        },
        "parameters": {
            "temperature": 0.7,  # 生成随机性：0（严谨）~1（灵活）
            "max_tokens": 1024,  # 最大生成令牌数（控制回复长度）
            "result_format": "message"  # 返回格式为消息体
        }
    }
    
    try:
        # 发送POST请求调用API
        response = requests.post(
            url=api_url,
            headers=headers,
            json=request_body,
            timeout=30  # 请求超时时间（秒）
        )
        response.raise_for_status()  # 抛出HTTP请求异常
        
        # 解析响应结果
        response_data = response.json()
        assistant_reply = response_data["output"]["choices"][0]["message"]["content"].strip()
        
        # 将机器人回复添加到聊天历史
        chat_history.append({
            "role": "assistant",
            "content": assistant_reply
        })
        
        return assistant_reply
    
    except Exception as e:
        return f"请求失败：{str(e)}"

def main():
    """
    主函数：运行聊天机器人，实现对话循环
    """
    print("=" * 50)
    print("  通义千问 聊天机器人（命令行版）")
    print("  输入 'exit' 或 'quit' 可退出程序")
    print("=" * 50 + "\n")
    
    try:
        # 构建请求头
        headers = get_tongyi_headers()
        
        # 初始化聊天历史
        chat_history = init_chat_history()
        
        # 对话循环
        while True:
            # 获取用户输入
            user_input = input("你：").strip()
            
            # 退出条件判断
            if user_input.lower() in ["exit", "quit"]:
                print("机器人：再见！欢迎下次再来聊天~")
                break
            
            # 空输入判断
            if not user_input:
                print("机器人：请输入有效内容哦~")
                continue
            
            # 调用API获取回复并打印
            print("机器人：正在思考中...")
            assistant_reply = chat_with_tongyi(headers, chat_history, user_input)
            print(f"机器人：{assistant_reply}\n")
    
    except Exception as e:
        print(f"程序异常：{str(e)}")

if __name__ == "__main__":
    main()