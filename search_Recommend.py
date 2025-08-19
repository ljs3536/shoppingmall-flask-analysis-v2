from collections import defaultdict
from datetime import datetime
from math import log
from pymongo import MongoClient
from config import Config

client = MongoClient(Config.MONGODB_URI)
db = client[Config.MONGODB_DB]

orderCol = db["order_products_logs"]
cartCol = db["cart_products_logs"]
reviewCol = db["review_products_logs"]

# format 동일하게 유지
def format_mongo_results(cursor, key_name="name", count_name="count"):
    return [{key_name: d["_id"], count_name: d["count"]} for d in cursor]

# 1. 단순 많이 팔린 상품 (Top 10)
def get_moreSellingProducts(sellerId=None):
    match_stage = {}
    if sellerId:
        match_stage["sellerId"] = sellerId

    pipeline = []
    if match_stage:
        pipeline.append({"$match": match_stage})

    pipeline += [
        {"$group": {"_id": "$productName", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    res = orderCol.aggregate(pipeline)
    return format_mongo_results(res)

# 2. 카테고리별 인기 상품 (Top 5 per category)
def get_popularProducts_category(sellerId=None):
    match_stage = {}
    if sellerId:
        match_stage["sellerId"] = sellerId

    pipeline = []
    if match_stage:
        pipeline.append({"$match": match_stage})

    pipeline += [
        {"$group": {
            "_id": {"category": "$productCategory", "product": "$productName"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}},
        {"$group": {
            "_id": "$_id.category",
            "products": {"$push": {"name": "$_id.product", "count": "$count"}}
        }},
        {"$project": {
            "products": {"$slice": ["$products", 5]}
        }}
    ]
    res = orderCol.aggregate(pipeline)
    result = []
    for doc in res:
        for p in doc["products"]:
            result.append({
                "category": doc["_id"],
                "name": p["name"],
                "count": p["count"]
            })
    return result

# 3. 장바구니에 많이 담긴 상품
def get_addedCartProducts(sellerId=None):
    match_stage = {"actionType": "ADD"}
    if sellerId:
        match_stage["sellerId"] = sellerId

    pipeline = [
        {"$match": match_stage},
        {"$group": {"_id": "$productName", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    res = cartCol.aggregate(pipeline)
    return format_mongo_results(res)

# 4. 리뷰 많은 + 평점 높은 상품
def get_highRatedProducts(sellerId=None):
    match_stage = {}
    if sellerId:
        match_stage["sellerId"] = sellerId

    pipeline = []
    if match_stage:
        pipeline.append({"$match": match_stage})

    pipeline += [
        {"$group": {
            "_id": "$productName",
            "avgRating": {"$avg": "$rating"},
            "reviewCount": {"$sum": 1}
        }},
        {"$sort": {"reviewCount": -1}},
        {"$limit": 10}
    ]
    res = reviewCol.aggregate(pipeline)
    return [
        {"name": doc["_id"], "avgRating": round(doc["avgRating"], 2), "reviewCount": doc["reviewCount"]}
        for doc in res
    ]

# 5. 최근 트렌디한 상품 (시간 가중치 기반)
def get_trendingProducts(sellerId=None):
    query = {}
    if sellerId:
        query["sellerId"] = sellerId

    cursor = orderCol.find(query, {"productName": 1, "timestamp": 1})
    scores = defaultdict(float)
    now = datetime.now()

    for doc in cursor:
        product = doc["productName"]
        ts = doc["timestamp"]
        if isinstance(ts, str):
            ts = datetime.strptime(ts, "%Y-%m-%d")
        days = (now - ts).days
        score = 1 / log(2 + days)
        scores[product] += score

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"name": name, "score": round(score, 2)} for name, score in sorted_scores[:10]]
