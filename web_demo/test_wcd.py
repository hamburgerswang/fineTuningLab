from dotenv import load_dotenv
import os
import weaviate
from weaviate.classes.init import Auth

# 加载环境变量
load_dotenv("api_keys.env")  # 👈 改这里

# 获取密钥
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# 连接到 Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url="https://ipu4fofq3cudvfcc1ek7a.c0.asia-southeast1.gcp.weaviate.cloud",
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
    additional_config=weaviate.config.AdditionalConfig(
        timeout=weaviate.config.Timeout(init=10)
    )
)

# 测试连接是否成功
if client.is_ready():
    print("✅ 成功连接到 Weaviate Cloud！")
    print(f"集群 URL: {weaviate_url}")
    print(f"API Key: {weaviate_api_key[:6]}...")  # 只显示前6位，保护隐私
else:
    print("❌ 连接失败，请检查网络、URL 或 API Key")

# 使用完后关闭连接
client.close()
