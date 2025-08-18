import os

class Config:
    ELASTICSEARCH_URI = os.getenv('ELASTICSEARCH_URI', 'http://localhost:9200')
    KAFKA_BOOTSTRAP_SERVERS = [
        'localhost:9092'
    ]
    KAFKA_CONSUMER_GROUP_ID = 'flask-group-test-01'
    KAFKA_TOPIC = 'model-train-topic'
    KAFKA_RESULT_TOPIC = 'model-train-result'