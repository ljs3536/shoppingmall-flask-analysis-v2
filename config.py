import os

class Config:
    # MongoDB 설정 (Elasticsearch → MongoDB)
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB = os.getenv("MONGODB_DB", "shoppingmall")

    # Kafka 설정 그대로 유지
    KAFKA_BOOTSTRAP_SERVERS = [
        'localhost:9092'
    ]
    KAFKA_CONSUMER_GROUP_ID = 'flask-group-test-01'
    KAFKA_TOPIC = 'model-train-topic'
    KAFKA_RESULT_TOPIC = 'model-train-result'
