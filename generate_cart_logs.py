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


def generate_cart_log(days=365*5):
    actions = []
    start_date = datetime.now() - timedelta(days=days)
    user_profiles = generate_user_profiles(n=1000)

    # userId를 키로 cart 상태 관리
    cart_state = {user["userId"]: set() for user in user_profiles}

    for day in range(days):
        log_date = start_date + timedelta(days=day)
        num_logs_per_day = random.randint(30, 200)
        for _ in range(num_logs_per_day):
            user = random.choice(user_profiles)
            user_id = user["userId"]
            product = random.choice(products)
            quantity = random.randint(1, 5)

            actionType = random.choice(["ADD", "UPDATE"])

            if actionType == "ADD":
                cart_state[user_id].add(product["name"])
            elif actionType == "UPDATE" and product["name"] not in cart_state[user_id]:
                continue  # UPDATE는 ADD된 것만 가능하니까 skip

            cart_log = {
                "_index": "cart_products-logs",
                "_source": {
                    "timestamp": log_date.strftime("%Y-%m-%d"),
                    "actionType": actionType,
                    "userId": user_id,
                    "userAge": user["age"],
                    "userRegion": user["region"],
                    "userGender": user["gender"],
                    "productName": product["name"],
                    "productPrice": product["price"],
                    "productCategory": product["category"],
                    "sellerId": product["sellerId"],
                    "productQuantity": quantity
                }
            }
            actions.append(cart_log)

            # ADD된 제품에 대해 30% 확률로 REMOVE
            if actionType == "ADD" and random.random() < 0.3:
                remove_log = cart_log.copy()
                remove_log["_source"] = cart_log["_source"].copy()  # 깊은 복사 필요
                remove_log["_source"]["actionType"] = "REMOVE"
                actions.append(remove_log)
                cart_state[user_id].discard(product["name"])  # 장바구니에서 제거

    helpers.bulk(es, actions)
    print(f"{len(actions)}개의 장바구니 로그가 생성되었습니다!")

# 실행 시
#generate_cart_log()
