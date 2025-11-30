# check_imported_data.py
from dotenv import load_dotenv
import os
import weaviate
from weaviate.classes.init import Auth

load_dotenv("api_keys.env")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url="https://ipu4fofq3cudvfcc1ek7a.c0.asia-southeast1.gcp.weaviate.cloud",
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
    additional_config=weaviate.config.AdditionalConfig(
        timeout=weaviate.config.Timeout(init=10)
    )
)
# 获取 Hotel 集合
hotel_collection = client.collections.get("Hotel")

# 方式 1: 获取前 5 条数据（不带向量）
response = hotel_collection.query.fetch_objects(limit=5)

print("✅ 已导入的前 5 条酒店数据：")
for obj in response.objects:
    print(obj.properties)  # 只打印属性，不包含向量/UUID

# 方式 2: 按 hotel_id 查询特定数据
# obj = hotel_collection.query.fetch_object_by_id("your-uuid-here")
# print(obj.properties)

client.close()