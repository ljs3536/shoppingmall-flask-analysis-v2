from flask import Flask, jsonify, request
from elasticsearch import Elasticsearch
from generate_cart_logs import generate_cart_log
from generate_order_logs import generate_order_log
from search_handler import get_yearly_sales, get_age_group_favorites, get_region_favorites, get_monthly_category_trend, get_gender_favorites
from train_order_product_model import train_predict_model_and_save
from predict_order_product_model import predict_quantity_pipeline
from train_recommend_product_model import train_recommend_model_and_save
from predict_recommend_product_model import predict_recommendation_pipeline
from search_Recommend import get_trendingProducts, get_addedCartProducts, get_moreSellingProducts, get_popularProducts_category, get_highRatedProducts
from prometheus_flask_exporter import PrometheusMetrics
from urllib.parse import unquote
import numpy as np
from kafka import KafkaConsumer
from kafka import KafkaProducer
import config
import threading
import json

app = Flask(__name__)
metrics = PrometheusMetrics(app, path='/metrics')
app.config.from_object(config.Config)

producer = KafkaProducer(
    bootstrap_servers=app.config['KAFKA_BOOTSTRAP_SERVERS'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def notify_model_trained(algo_name, model_type, status, log_id):
    producer.send(app.config['KAFKA_RESULT_TOPIC'], {
        'algo_name': algo_name,
        'model_type': model_type,
        'status': status,  # e.g., 'success' or 'fail'
        'log_id': log_id
    })
    producer.flush()

def kafka_consumer_job():
    consumer = KafkaConsumer(
        app.config['KAFKA_TOPIC'],
        bootstrap_servers=app.config['KAFKA_BOOTSTRAP_SERVERS'],
        group_id=app.config['KAFKA_CONSUMER_GROUP_ID'],
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        enable_auto_commit=True
    )
    for message in consumer:
        data = message.value
        algo_name = data.get('algo_name')
        uri = data.get('uri')
        log_id = data.get('log_id')
        print(f"Kafka 메시지 수신: {algo_name}")
        print(f"uri: {uri}")
        if "predict" in uri:
            predict_train_model(algo_name, log_id)
        else:
            recommend_train_model(algo_name, log_id)

def predict_train_model(algo_name, log_id):
    print(f"모델 학습 시작: {algo_name}")
    result = train_predict_model_and_save(algo_name)
    print(result)
    notify_model_trained(algo_name, 'predict', 'success', log_id)

def recommend_train_model(algo_name, log_id):
    print(f"모델 학습 시작: {algo_name}")
    result = train_recommend_model_and_save(algo_name)
    print(result)
    notify_model_trained(algo_name, 'recommend', 'success', log_id)

@app.route("/")
def index():
    return "Hello, Prometheus!"

# Elasticsearch 연결 (Docker 컨테이너에서 실행 중일 경우)
from config import Config
es = Elasticsearch(Config.ELASTICSEARCH_URI)

@app.route("/search", methods=["GET"])
def search_logs():
    keyword = request.args.get("keyword", "")
    index_name = "access-log"  # 로그가 저장된 인덱스 이름

    query = {
        "query": {
            "match": {
                "userId": keyword
            }
        }
    }

    res = es.search(index=index_name, body=query)
    return jsonify(res["hits"]["hits"])

# /generate/cart 엔드포인트
@app.route('/generate/cart', methods=['GET'])
def generate_cart_logs():
    generate_cart_log()
    return jsonify({"message": "Cart logs generated!"})

# /generate/order 엔드포인트
@app.route('/generate/order', methods=['GET'])
def generate_order_logs():
    generate_order_log()
    return jsonify({"message": "Order logs generated!"})

@app.route("/search/products/years", methods=["GET"])
def get_sales_by_year():
    year = request.args.get("year")
    if not year:
        return jsonify({"error": "year parameter required"}), 400

    data = get_yearly_sales(year)
    return jsonify(data)

@app.route("/search/products/age", methods=["GET"])
def age_group_favorites():
    return jsonify(get_age_group_favorites())

@app.route("/search/products/region", methods=["GET"])
def region_favorites():
    return jsonify(get_region_favorites())

@app.route("/search/products/trend", methods=["GET"])
def monthly_trend():
    return jsonify(get_monthly_category_trend())

@app.route("/search/products/gender", methods=["GET"])
def gender_favorites():
    return jsonify(get_gender_favorites())

@app.route("/search/products/moreSelling", methods=["GET"])
def more_selling():
    seller_id = request.args.get("sellerId")
    return jsonify(get_moreSellingProducts(seller_id))

@app.route("/search/products/popularByCategory", methods=["GET"])
def popular_by_category():
    seller_id = request.args.get("sellerId")
    return jsonify(get_popularProducts_category(seller_id))

@app.route("/search/products/addedCart", methods=["GET"])
def added_cart():
    seller_id = request.args.get("sellerId")
    return jsonify(get_addedCartProducts(seller_id))

@app.route("/search/products/highRated", methods=["GET"])
def high_rated():
    seller_id = request.args.get("sellerId")
    return jsonify(get_highRatedProducts(seller_id))

@app.route("/search/products/trending", methods=["GET"])
def trending():
    seller_id = request.args.get("sellerId")
    return jsonify(get_trendingProducts(seller_id))

@app.route("/predict/train", methods=["POST"])
def train_predict_model():
    algo_name = request.json.get("algo_name")
    print(algo_name)
    result = train_predict_model_and_save(algo_name)
    return jsonify(result)

@app.route("/predict/product", methods=["GET"])
def predict_product_quantity():
    # URL 인코딩된 productName 파라미터 가져오기
    encoded_product_name = request.args.get('productName', '')

    # URL 디코딩하여 원래 한글로 변환
    product_name = unquote(encoded_product_name)
    algo_name = request.args.get("algo", default="linear")

    print(product_name, " : " , algo_name)
    if not product_name:
        return jsonify({"error": "productName parameter is required"}), 400

    result = predict_quantity_pipeline(product_name, algo_name)

    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        else:
            return obj

    # 결과 변환
    result = convert_numpy_types(result)

    return jsonify(result)

@app.route("/recommend/train", methods=["POST"])
def train_recommend_model():
    algo_name = request.json.get("algo_name")
    print(algo_name)
    result = train_recommend_model_and_save(algo_name)
    return jsonify(result)

@app.route("/recommend/product", methods=["POST"])
def predict_product_recommend():
    user_info = request.json.get("user_info")
    algo_name = request.args.get("algo", default="linear")

    if not user_info:
        return jsonify({"error": "user_info parameter is required"}), 400

    result = predict_recommendation_pipeline(user_info, algo_name)
    return jsonify(result)

if __name__ == "__main__":
    kafka_thread = threading.Thread(target=kafka_consumer_job)
    kafka_thread.daemon = True
    kafka_thread.start()

    app.run(host='0.0.0.0', port=6000, debug=False)
