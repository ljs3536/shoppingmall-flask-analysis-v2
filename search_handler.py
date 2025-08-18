from elasticsearch import Elasticsearch
from collections import defaultdict
import pandas as pd

from config import Config
es = Elasticsearch(Config.ELASTICSEARCH_URI)
index_name = "order_products-logs"

def get_yearly_sales(year: str):
    body = {
        "size": 0,  # 실데이터는 가져오지 않고 aggregation만
        "query": {
            "range": {
                "timestamp": {
                    "gte": f"{year}-01-01",
                    "lte": f"{year}-12-31"
                }
            }
        },
        "aggs": {
            "products": {
                "terms": {
                    "field": "productName",
                    "size": 1000  # 연도 내 상품 종류 수가 많으면 늘리기
                },
                "aggs": {
                    "total_quantity": {
                        "sum": {
                            "field": "productQuantity"
                        }
                    }
                }
            }
        }
    }

    res = es.search(index=index_name, body=body)
    sales_summary = {}
    for bucket in res["aggregations"]["products"]["buckets"]:
        product = bucket["key"]
        quantity = int(bucket["total_quantity"]["value"])
        sales_summary[product] = quantity

    return sales_summary

def get_age_group_favorites():
    body = {
        "size": 0,
        "aggs": {
            "age_groups": {
                "terms": {
                    "script": {
                        "source": """
                            if (doc['userAge'].size() == 0) {
                                return null;
                            } else {
                                return Math.floor(doc['userAge'].value / 10) * 10;
                            }
                        """,
                        "lang": "painless"
                    },
                    "size": 10
                },
                "aggs": {
                    "top_products": {
                        "terms": {
                            "field": "productName",
                            "size": 1,
                            "order": {"total_quantity": "desc"}
                        },
                        "aggs": {
                            "total_quantity": {
                                "sum": {
                                    "field": "productQuantity"
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    res = es.search(index=index_name, body=body)
    result = []
    for bucket in res['aggregations']['age_groups']['buckets']:
        age = bucket['key']

        # top_products buckets가 비어있는지 확인
        if not bucket['top_products']['buckets']:
            continue

        top_product = bucket['top_products']['buckets'][0]
        result.append({
            "ageGroup": age,
            "productName": top_product['key'],
            "productQuantity": int(top_product['total_quantity']['value'])
        })
    print(result)
    return result

def get_gender_favorites():
    body = {
        "size": 0,
        "aggs": {
            "gender_groups": {
                "terms": {
                    "field": "userGender",  # 예: "M", "F"
                    "size": 10
                },
                "aggs": {
                    "top_products": {
                        "terms": {
                            "field": "productName",
                            "size": 1,
                            "order": {"total_quantity": "desc"}
                        },
                        "aggs": {
                            "total_quantity": {
                                "sum": {
                                    "field": "productQuantity"
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    res = es.search(index=index_name, body=body)
    result = []
    for bucket in res['aggregations']['gender_groups']['buckets']:
        gender = bucket['key']
        top_product = bucket['top_products']['buckets'][0]
        result.append({
            "userGender": gender,
            "productName": top_product['key'],
            "productQuantity": int(top_product['total_quantity']['value'])
        })

    return result

def get_region_favorites():
    body = {
        "size": 0,
        "aggs": {
            "region_groups": {
                "terms": {
                    "field": "userRegion",
                    "size": 20
                },
                "aggs": {
                    "top_products": {
                        "terms": {
                            "field": "productName",
                            "size": 1,
                            "order": {"total_quantity": "desc"}
                        },
                        "aggs": {
                            "total_quantity": {
                                "sum": {
                                    "field": "productQuantity"
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    res = es.search(index=index_name, body=body)
    result = []
    for bucket in res['aggregations']['region_groups']['buckets']:
        region = bucket['key']
        top_product = bucket['top_products']['buckets'][0]
        result.append({
            "userRegion": region,
            "productName": top_product['key'],
            "productQuantity": int(top_product['total_quantity']['value'])
        })

    return result

def get_monthly_category_trend():
    body = {
        "size": 0,
        "aggs": {
            "monthly": {
                "date_histogram": {
                    "field": "timestamp",
                    "calendar_interval": "month",
                    "format": "yyyy-MM"
                },
                "aggs": {
                    "category": {
                        "terms": {
                            "field": "productCategory",
                            "size": 20
                        },
                        "aggs": {
                            "total_quantity": {
                                "sum": {
                                    "field": "productQuantity"
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    res = es.search(index=index_name, body=body)
    result = []
    for month_bucket in res['aggregations']['monthly']['buckets']:
        month = month_bucket['key_as_string']
        for cat_bucket in month_bucket['category']['buckets']:
            result.append({
                "month": month,
                "productCategory": cat_bucket['key'],
                "productQuantity": int(cat_bucket['total_quantity']['value'])
            })

    return sorted(result, key=lambda x: x["month"])