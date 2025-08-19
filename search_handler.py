from pymongo import MongoClient
from config import Config
from datetime import datetime

client = MongoClient(Config.MONGODB_URI)
db = client[Config.MONGODB_DB]

orderCol = db["order_products_logs"]

# 1. 연도별 판매량
def get_yearly_sales(year: str):
    start = datetime.strptime(f"{year}-01-01", "%Y-%m-%d")
    end = datetime.strptime(f"{year}-12-31", "%Y-%m-%d")

    pipeline = [
        {"$match": {"timestamp": {"$gte": start, "$lte": end}}},
        {"$group": {"_id": "$productName", "total_quantity": {"$sum": "$productQuantity"}}},
        {"$sort": {"total_quantity": -1}}
    ]

    res = orderCol.aggregate(pipeline)
    sales_summary = {doc["_id"]: doc["total_quantity"] for doc in res}
    return sales_summary


# 2. 연령대별 인기 상품
def get_age_group_favorites():
    pipeline = [
        {"$match": {"userAge": {"$ne": None}}},
        {"$project": {
            "ageGroup": {"$multiply": [{"$floor": {"$divide": ["$userAge", 10]}}, 10]},
            "productName": 1,
            "productQuantity": 1
        }},
        {"$group": {"_id": {"ageGroup": "$ageGroup", "productName": "$productName"},
                    "total_quantity": {"$sum": "$productQuantity"}}},
        {"$sort": {"total_quantity": -1}},
        {"$group": {"_id": "$_id.ageGroup",
                    "top_product": {"$first": {"name": "$_id.productName", "quantity": "$total_quantity"}}}}
    ]

    res = orderCol.aggregate(pipeline)
    return [{"ageGroup": doc["_id"], "productName": doc["top_product"]["name"], "productQuantity": doc["top_product"]["quantity"]} for doc in res]


# 3. 성별 인기 상품
def get_gender_favorites():
    pipeline = [
        {"$match": {"userGender": {"$ne": None}}},
        {"$group": {"_id": {"gender": "$userGender", "productName": "$productName"},
                    "total_quantity": {"$sum": "$productQuantity"}}},
        {"$sort": {"total_quantity": -1}},
        {"$group": {"_id": "$_id.gender",
                    "top_product": {"$first": {"name": "$_id.productName", "quantity": "$total_quantity"}}}}
    ]

    res = orderCol.aggregate(pipeline)
    return [{"userGender": doc["_id"], "productName": doc["top_product"]["name"], "productQuantity": doc["top_product"]["quantity"]} for doc in res]


# 4. 지역별 인기 상품
def get_region_favorites():
    pipeline = [
        {"$match": {"userRegion": {"$ne": None}}},
        {"$group": {"_id": {"region": "$userRegion", "productName": "$productName"},
                    "total_quantity": {"$sum": "$productQuantity"}}},
        {"$sort": {"total_quantity": -1}},
        {"$group": {"_id": "$_id.region",
                    "top_product": {"$first": {"name": "$_id.productName", "quantity": "$total_quantity"}}}}
    ]

    res = orderCol.aggregate(pipeline)
    return [{"userRegion": doc["_id"], "productName": doc["top_product"]["name"], "productQuantity": doc["top_product"]["quantity"]} for doc in res]


# 5. 월별 카테고리 트렌드
def get_monthly_category_trend():
    pipeline = [
        {"$project": {
            "month": {"$dateToString": {"format": "%Y-%m", "date": "$timestamp"}},
            "productCategory": 1,
            "productQuantity": 1
        }},
        {"$group": {"_id": {"month": "$month", "category": "$productCategory"},
                    "total_quantity": {"$sum": "$productQuantity"}}},
        {"$sort": {"_id.month": 1, "total_quantity": -1}}
    ]

    res = orderCol.aggregate(pipeline)
    result = []
    for doc in res:
        result.append({
            "month": doc["_id"]["month"],
            "productCategory": doc["_id"]["category"],
            "productQuantity": doc["total_quantity"]
        })
    return result
