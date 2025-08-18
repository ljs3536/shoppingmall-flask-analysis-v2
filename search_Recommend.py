from elasticsearch import Elasticsearch
from collections import defaultdict
from datetime import datetime
from math import log

from config import Config
es = Elasticsearch(Config.ELASTICSEARCH_URI)

orderEs = "order_products-logs"
cartEs = "cart_products-logs"
reviewEs = "review_products-logs"

def format_es_buckets(bucket_data):
    return [
        {"name": b["key"], "count": b["doc_count"]}
        for b in bucket_data
    ]

# 1. 단순 많이 팔린 상품 (Top 10)
def get_moreSellingProducts(sellerId=None):
    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": []
            }
        },
        "aggs": {
            "top_products": {
                "terms": {
                    "field": "productName.keyword",
                    "size": 10
                }
            }
        }
    }
    if sellerId:
        body["query"]["bool"]["filter"].append({
            "term": {
                "sellerId": sellerId
            }
        })
    else:
        body.pop("query")  # sellerId 없으면 query 제거

    res = es.search(index=orderEs, body=body)
    return format_es_buckets(res['aggregations']['top_products']['buckets'])


# 2. 카테고리별 인기 상품 (Top 5 per category)
def get_popularProducts_category(sellerId=None):
    query_filter = []
    if sellerId:
        query_filter.append({
            "term": {
                "sellerId": sellerId
            }
        })

    body = {
        "size": 0,
        "aggs": {
            "by_category": {
                "terms": {
                    "field": "productCategory.keyword"
                },
                "aggs": {
                    "top_products": {
                        "terms": {
                            "field": "productName.keyword",
                            "size": 5
                        }
                    }
                }
            }
        }
    }

    if sellerId:
        body["query"] = {
            "bool": {
                "filter": query_filter
            }
        }
    res = es.search(index=orderEs, body=body)
    result = []
    for cat in res['aggregations']['by_category']['buckets']:
        category_name = cat['key']
        for product in cat['top_products']['buckets']:
            result.append({
                "category": category_name,
                "name": product["key"],
                "count": product["doc_count"]
            })
    return result


# 3. 장바구니에 많이 담긴 상품 (잠재 인기)
def get_addedCartProducts(sellerId=None):
    query_filter = [{"term": {"actionType": "ADD"}}]
    if sellerId:
        query_filter.append({"term": {"sellerId": sellerId}})

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": query_filter
            }
        },
        "aggs": {
            "popular_cart_items": {
                "terms": {
                    "field": "productName",
                    "size": 10
                }
            }
        }
    }

    res = es.search(index=cartEs, body=body)
    return format_es_buckets(res['aggregations']['popular_cart_items']['buckets'])


# 4. 리뷰 많은 + 평점 높은 상품
def get_highRatedProducts(sellerId=None):
    query_filter = []
    if sellerId:
        query_filter.append({
            "term": {
                "sellerId": sellerId
            }
        })

    body = {
        "size": 0,
        "aggs": {
            "top_reviews": {
                "terms": {
                    "field": "productName.keyword",
                    "size": 10
                },
                "aggs": {
                    "avg_rating": {
                        "avg": {
                            "field": "rating"
                        }
                    },
                    "review_count": {
                        "value_count": {
                            "field": "rating"
                        }
                    }
                }
            }
        }
    }

    if sellerId:
        body["query"] = {
            "bool": {
                "filter": query_filter
            }
        }
    res = es.search(index=reviewEs, body=body)
    buckets = res['aggregations']['top_reviews']['buckets']
    
    result = []
    for b in buckets:
        result.append({
            "name": b['key'],
            "avgRating": round(b['avg_rating']['value'], 2),
            "reviewCount": b['review_count']['value']
        })
    return result


# 5. 최근 트렌디한 상품 (시간 가중치 기반)
def get_trendingProducts(sellerId=None):
    query_filter = []
    if sellerId:
        query_filter.append({
            "term": {
                "sellerId": sellerId
            }
        })

    body = {
        "size": 10000,
        "_source": ["productName", "timestamp"],
        "query": {
            "bool": {
                "filter": query_filter or [{"match_all": {}}]
            }
        }
    }

    res = es.search(index=orderEs, body=body)
    scores = defaultdict(float)
    now = datetime.now()

    for hit in res['hits']['hits']:
        product = hit['_source']['productName']
        ts = hit['_source']['timestamp']
        date = datetime.strptime(ts, "%Y-%m-%d")
        days = (now - date).days
        score = 1 / log(2 + days)
        scores[product] += score

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"name": name, "score": round(score, 2)} for name, score in sorted_scores[:10]]

