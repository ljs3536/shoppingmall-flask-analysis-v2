from datetime import datetime, timedelta
import random
from elasticsearch import Elasticsearch, helpers

from config import Config
es = Elasticsearch(Config.ELASTICSEARCH_URI)

products = [  # 동일한 제품 리스트 유지
    {"name": "노트북1", "price": 11200000, "category": "전자제품", "sellerId": "testseller1"},
    {"name": "스마트폰1", "price": 1900000, "category": "전자제품", "sellerId": "testseller1"},
    {"name": "헤드폰1", "price": 2150000, "category": "전자제품", "sellerId": "testseller1"},
    {"name": "노트북2", "price": 1200000, "category": "전자제품", "sellerId": "testseller1"},
    {"name": "스마트폰2", "price": 900000, "category": "전자제품", "sellerId": "testseller1"},
    {"name": "헤드폰2", "price": 50000, "category": "전자제품", "sellerId": "testseller1"},
    {"name": "볼펜1", "price": 2000, "category": "생활용품", "sellerId": "testseller1"},
    {"name": "볼펜2", "price": 4000, "category": "생활용품", "sellerId": "testseller1"},
    {"name": "볼펜3", "price": 12000, "category": "생활용품", "sellerId": "testseller1"},
    {"name": "가위1", "price": 3000, "category": "생활용품", "sellerId": "testseller1"},
    {"name": "물티슈1", "price": 120000, "category": "생활용품", "sellerId": "testseller1"},
    {"name": "물티슈2", "price": 20000, "category": "생활용품", "sellerId": "testseller11"},
    {"name": "물티슈3", "price": 320000, "category": "생활용품", "sellerId": "testseller11"},
    {"name": "휴지1", "price": 50000, "category": "생활용품", "sellerId": "testseller12"},
    {"name": "운동화1", "price": 120000, "category": "패션", "sellerId": "testseller13"},
    {"name": "청바지1", "price": 50000, "category": "패션", "sellerId": "testseller13"},
    {"name": "운동화2", "price": 20000, "category": "패션", "sellerId": "testseller13"},
    {"name": "청바지2", "price": 150000, "category": "패션", "sellerId": "testseller13"},
    {"name": "코트1", "price": 2000000, "category": "패션", "sellerId": "testseller13"},
    {"name": "코트2", "price": 250000, "category": "패션", "sellerId": "testseller13"},
    {"name": "코트3", "price": 8000000, "category": "패션", "sellerId": "testseller13"},
    {"name": "가디건1", "price": 100000, "category": "패션", "sellerId": "testseller13"},
    {"name": "가디건2", "price": 150000, "category": "패션", "sellerId": "testseller13"},
    {"name": "가디건3", "price": 80000, "category": "패션", "sellerId": "testseller13"},
    {"name": "핸드크림1", "price": 12000, "category": "화장품", "sellerId": "testseller12"},
    {"name": "선크림1", "price": 30000, "category": "화장품", "sellerId": "testseller12"},
    {"name": "핸드크림2", "price": 50000, "category": "화장품", "sellerId": "testseller12"},
    {"name": "선크림2", "price": 22000, "category": "화장품", "sellerId": "testseller12"},
    {"name": "핸드크림3", "price": 56000, "category": "화장품", "sellerId": "testseller12"},
    {"name": "선크림3", "price": 10000, "category": "화장품", "sellerId": "testseller12"},
]

positive_reviews = [
    "정말 만족스러워요!", "좋은 제품이에요", "다시 구매하고 싶어요", "추천합니다", "가성비 최고!", "또 구매했어요"
]

negative_reviews = [
    "별로에요", "품질이 기대 이하에요", "다시는 안살래요", "돈이 아까워요", "실망했어요", "이런상품 팔지마라"
]


def generate_user_profiles(n=1000):
    genders = ["남", "여"]
    regions = ["서울", "부산", "인천", "대전", "광주", "대구", "울산", "강릉", "전주", "천안", "세종"]

    users = []
    for i in range(1, n + 1):
        user = {
            "userId": f"testuser{i}",  # user0001 ~ user1000
            "age": random.randint(10, 80),
            "region": random.choice(regions),
            "gender": random.choice(genders)
        }
        users.append(user)
    return users


def generate_order_log(days=365*5):
    order_actions = []
    review_actions = []
    start_date = datetime.now() - timedelta(days=days)
    user_profiles = generate_user_profiles(n=1000)
    for day in range(days):
        log_date = start_date + timedelta(days=day)
        num_logs_per_day = random.randint(30, 200)

        for _ in range(num_logs_per_day):
            user = random.choice(user_profiles)
            product = random.choice(products)
            quantity = random.randint(1, 5)
            order_type = random.choice(["CART", "DIRECT"])

            order_doc = {
                "timestamp": log_date.strftime("%Y-%m-%d"),
                "orderType": order_type,
                "userId": user["userId"],
                "userAge": user["age"],
                "userRegion": user["region"],
                "userGender": user["gender"],
                "productName": product["name"],
                "productPrice": product["price"],
                "productCategory": product["category"],
                "sellerId": product["sellerId"],
                "productQuantity": float(quantity)
            }

            order_actions.append({
                "_index": "order_products-logs",
                "_source": order_doc
            })

            # 리뷰 생성 확률: 70%
            if random.random() < 0.7:
                rating = random.randint(1, 5)
                if rating >= 4:
                    description = random.choice(positive_reviews)
                else:
                    description = random.choice(negative_reviews)

                review_doc = {
                    "timestamp": log_date.strftime("%Y-%m-%d"),
                    "userId": user["userId"],
                    "userAge": user["age"],
                    "userRegion": user["region"],
                    "userGender": user["gender"],
                    "productName": product["name"],
                    "productPrice": product["price"],
                    "productCategory": product["category"],
                    "productQuantity": float(quantity),
                    "sellerId": product["sellerId"],
                    "rating": float(rating),
                    "description": description
                }

                review_actions.append({
                    "_index": "review_products-logs",
                    "_source": review_doc
                })

    helpers.bulk(es, order_actions)
    helpers.bulk(es, review_actions)
    print(f"{len(order_actions)}개의 주문 로그 생성 완료")
    print(f"{len(review_actions)}개의 리뷰 로그 생성 완료")



# 실행 시
#generate_order_log()
