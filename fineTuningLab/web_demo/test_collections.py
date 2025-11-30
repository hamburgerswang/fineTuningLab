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

print("当前所有集合（Collections）:")
collections = client.collections.list_all()
for name in collections:
    print(f" - {name}")

client.close()