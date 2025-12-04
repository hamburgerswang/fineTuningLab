import json

from dotenv import load_dotenv
load_dotenv("api_keys.env")
import os
import requests
import json
from tqdm import tqdm
import weaviate
from weaviate.classes.init import Auth



# 计算RRF分数
# 它根据文档在各个搜索结果列表中的排名位置计算分数，将高排名位置给予更高的权重。
def rrf(rankings, k=60):
    if not isinstance(rankings, list):
        raise ValueError("Rankings should be a list.")
    scores = dict()
    for ranking in rankings:
        if not ranking:  # 如果ranking为空，跳过它
            continue
        for i, doc in enumerate(ranking):
            if not isinstance(doc, dict):
                raise ValueError("Each item should be dict type.")
            doc_id = doc.get('hotel_id', None)
            if doc_id is None:
                raise ValueError("Each item should have 'hotel_id' key.")
            if doc_id not in scores:
                scores[doc_id] = (0, doc)
            scores[doc_id] = (scores[doc_id][0] + 1 / (k + i), doc)

    sorted_scores = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return [item[1] for item in sorted_scores]


class HotelDB():
    def __init__(self):
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url="https://ipu4fofq3cudvfcc1ek7a.c0.asia-southeast1.gcp.weaviate.cloud",
            auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
            headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY"),
                "X-HuggingFace-Api-Key": os.getenv("HUGGINGFACE_API_KEY")},
            additional_config=weaviate.config.AdditionalConfig(
                timeout=weaviate.config.Timeout(init=10)
            )
        )
        self.client = client

    def insert(self):
        """用 v4 方式创建 Hotel Collection 并导入数据"""
        from weaviate.classes.config import Configure, Property, DataType, Tokenization

        collection_name = "Hotel"

        # 使用上下文管理器确保连接关闭
        with self.client as client:
            # 删除已存在的 Collection
            if client.collections.exists(collection_name):
                print(f"⚠️ Collection '{collection_name}' 已存在，正在删除...")
                client.collections.delete(collection_name)

            # 创建新 Collection
            client.collections.create(
                name=collection_name,
                description="hotel info",
                # vectorizer_config=Configure.Vectorizer.text2vec_huggingface(
                #     model="sentence-transformers/all-MiniLM-L6-v2",  # 免费、轻量、中文可用
                #     wait_for_model=False,
                #     use_gpu=False,
                #     vectorize_collection_name=False,
                # ),
                vectorizer_config=Configure.Vectorizer.text2vec_huggingface(
                    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 多语言，支持中文
                    wait_for_model=False,
                    use_gpu=False,
                    vectorize_collection_name=False,
                ),
                properties=[
                    # hotel_id
                    Property(
                        name="hotel_id",
                        data_type=DataType.INT,
                        description="id of hotel"
                    ),
                    # _name（用于 BM25 搜索）
                    # BM25 逻辑是一种关键词（Keyword）匹配的评分算法，主要用于信息检索领域，是现代搜索引擎和文本数据库（如 Elasticsearch、Lucene）中广泛使用的一种相关性评分函数。
                    Property(
                        name="_name",
                        data_type=DataType.TEXT,
                        description="name of hotel (tokenized for search)",
                        index_filterable=True,
                        index_searchable=True,
                        tokenization=Tokenization.WHITESPACE,  # ✅ 修复点1
                        # skip_vectorization=True,
                    ),
                    # name（原始值）
                    Property(
                        name="name",
                        data_type=DataType.TEXT,
                        description="type of hotel",
                        # skip_vectorization=True,
                    ),
                    # type
                    Property(
                        name="type",
                        data_type=DataType.TEXT,
                        description="type of hotel",
                        # skip_vectorization=True,
                    ),
                    # _address（用于 BM25 搜索）
                    Property(
                        name="_address",
                        data_type=DataType.TEXT,
                        description="address of hotel (tokenized for search)",
                        index_filterable=True,
                        index_searchable=True,
                        tokenization=Tokenization.WHITESPACE,  # ✅ 修复点1
                        # skip_vectorization=True,
                    ),
                    # address（原始值）
                    Property(
                        name="address",
                        data_type=DataType.TEXT,
                        description="type of hotel",
                        # skip_vectorization=True,
                    ),
                    # subway
                    Property(
                        name="subway",
                        data_type=DataType.TEXT,
                        description="nearby subway",
                        # skip_vectorization=True,
                    ),
                    # phone
                    Property(
                        name="phone",
                        data_type=DataType.TEXT,
                        description="phone of hotel",
                        # skip_vectorization=True,
                    ),
                    # price
                    Property(
                        name="price",
                        data_type=DataType.NUMBER,
                        description="price of hotel"
                    ),
                    # rating
                    Property(
                        name="rating",
                        data_type=DataType.NUMBER,
                        description="rating of hotel"
                    ),
                    # facilities（唯一被向量化的文本字段）
                    Property(
                        name="facilities",
                        data_type=DataType.TEXT,
                        description="facilities provided",
                        index_filterable=True,
                        index_searchable=True,
                        skip_vectorization=False,  # 允许 OpenAI 向量化
                    ),
                ]
            )
            print(f"✅ Collection '{collection_name}' 创建成功")

            url = "https://raw.githubusercontent.com/hamburgerswang/hotel-chatbot/main/data/hotel.json"
            if not os.path.exists("hotel.json"):
                print("📥 正在下载 hotel.json...")
                try:
                    response = requests.get(url, timeout=30)  # 增加超时时间
                    response.raise_for_status()
                    with open("hotel.json", "w", encoding="utf-8") as f:
                        json.dump(response.json(), f, ensure_ascii=False, indent=2)
                    print("✅ 下载完成")
                except Exception as e:
                    print(f"❌ 下载失败: {e}")
                    return  # 如果下载失败，提前退出，避免后续操作
            else:
                print("📁 hotel.json 已存在")

            with open("hotel.json", "r", encoding="utf-8") as f:
                hotels = json.load(f)

            # 批量导入数据
            collection = client.collections.get(collection_name)
            print(f"📤 正在导入 {len(hotels)} 条酒店数据...")

            with collection.batch.dynamic() as batch:
                for hotel in tqdm(hotels, desc="导入进度"):
                    batch.add_object(
                        properties=hotel,
                        uuid=weaviate.util.generate_uuid5(hotel, collection_name)
                    )

            # 检查失败对象
            if collection.batch.failed_objects:
                print(f"⚠️ 导入失败数量: {len(collection.batch.failed_objects)}")
                print("第一个失败对象错误:", collection.batch.failed_objects[0].message)
            else:
                print("✅ 所有数据导入成功！")

    def search(self, dsl, name="Hotel", limit=1):
        # 清理 DSL
        dsl = {k: v for k, v in dsl.items() if v is not None}
        _limit = limit + 10
        output_fields = ["hotel_id", "name", "type", "address", "phone", "subway", "facilities", "price", "rating"]

        collection = self.client.collections.get(name)

        # === 1. 构建 filters (v4) ===
        from weaviate.classes.query import Filter
        filters = None

        if "type" in dsl:
            filters = Filter.by_property("type").equal(dsl["type"])
        if "price_range_lower" in dsl:
            f = Filter.by_property("price").greater_than(dsl["price_range_lower"])
            filters = f if filters is None else filters & f
        if "price_range_upper" in dsl:
            f = Filter.by_property("price").less_than(dsl["price_range_upper"])
            filters = f if filters is None else filters & f
        if "rating_range_lower" in dsl:
            f = Filter.by_property("rating").greater_than(dsl["rating_range_lower"])
            filters = f if filters is None else filters & f
        if "rating_range_upper" in dsl:
            f = Filter.by_property("rating").less_than(dsl["rating_range_upper"])
            filters = f if filters is None else filters & f

        candidates = []

        # === 2. 向量搜索 (facilities) ===
        if "facilities" in dsl and dsl["facilities"]:
            query_text = "酒店提供：" + "，".join(dsl["facilities"])
            res = collection.query.near_text(
                query=query_text,
                limit=_limit,
                filters=filters,
                return_properties=output_fields
            )
            candidates = [obj.properties for obj in res.objects]

        # === 3. 关键词搜索 (name) ===
        elif "name" in dsl and dsl["name"]:
            import re
            clean_name = " ".join(re.findall(r"[\w\-]+", dsl["name"]))
            res = collection.query.bm25(
                query=clean_name,
                query_properties=["_name"],
                limit=_limit,
                filters=filters,
                return_properties=output_fields
            )
            candidates = [obj.properties for obj in res.objects]

        # === 4. 关键词搜索 (address) ===
        elif "address" in dsl and dsl["address"]:
            import re
            clean_addr = " ".join(re.findall(r"[\w\-]+", dsl["address"]))
            res = collection.query.bm25(
                query=clean_addr,
                query_properties=["_address"],
                limit=_limit,
                filters=filters,
                return_properties=output_fields
            )
            candidates = [obj.properties for obj in res.objects]

        # === 5. 纯结构化过滤 ===
        else:
            res = collection.query.fetch_objects(
                limit=_limit,
                filters=filters,
                return_properties=output_fields
            )
            candidates = [obj.properties for obj in res.objects]

        # === 6. 排序 ===
        if "sort.slot" in dsl:
            reverse = dsl.get("sort.ordering") == "descend"
            slot = dsl["sort.slot"]
            candidates = sorted(candidates, key=lambda x: x.get(slot, 0), reverse=reverse)

        # === 7. name 后过滤（子串匹配）===
        if "name" in dsl:
            filtered = []
            for r in candidates:
                if dsl["name"] in r.get("name", ""):
                    filtered.append(r)
            candidates = filtered

        return candidates[:limit]


if __name__ == "__main__":
    db = HotelDB()
    try:
        # insert
        db.insert()
        print("✅ 数据导入完成！")
        # 你的逻辑，比如 db.search(...)
        # result = db.search({"facilities": ["wifi"]}, limit=3)
        # print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        # 确保连接被关闭
        db.client.close()
