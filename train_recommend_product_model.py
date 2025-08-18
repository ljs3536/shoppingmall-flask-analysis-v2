import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
import pickle, os
from sklearn.neighbors import NearestNeighbors
from gensim.models import Word2Vec
from config import Config
from pymongo import MongoClient
model_dir = "/app/model_storage/recommend"
os.makedirs(model_dir, exist_ok=True)

mongo_client = MongoClient("mongodb://root:rootpass@localhost:27017/")
db = mongo_client["smartmall"]

def fetch_all_es_data(index_name, es, scroll='2m', size=1000):
    all_data = []
    page = es.search(index=index_name, scroll=scroll, size=size, body={"query": {"match_all": {}}})
    sid = page['_scroll_id']
    hits = page['hits']['hits']
    all_data.extend(hits)

    while hits:
        page = es.scroll(scroll_id=sid, scroll=scroll)
        sid = page['_scroll_id']
        hits = page['hits']['hits']
        all_data.extend(hits)

    return [doc['_source'] for doc in all_data]


def train_recommend_model_and_save(algo_name: str):
    recommendation_algos = ["content", "collaborative", "svd", "xgb_classifier", "knn","item2vec"]
    if algo_name in recommendation_algos:
        return train_recommendation_model(algo_name)
    else:
        raise ValueError(f"Unsupported or invalid algorithm: {algo_name}")


def train_recommendation_model(algo_name: str):
    es = Elasticsearch(Config.ELASTICSEARCH_URI)
    index_name = "order_products-logs"
    data = fetch_all_es_data(index_name, es)
    df = pd.DataFrame(data)

    # 컬럼명 정리
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["userId"] = df["userId"].astype(str)
    df["productName"] = df["productName"].astype(str)

    # 필드명 매칭
    df.rename(columns={
        "userAge": "age",
        "userGender": "gender",
        "userRegion": "region",
        "productName": "product"
    }, inplace=True)

    user_features = df[["userId", "region", "age", "gender"]].drop_duplicates()

    model_path = os.path.join(model_dir, f"model_{algo_name}.pkl")

    if algo_name == "content":

        # 1. 범주형 수치화 (user 정보)
        user_features_encoded = pd.get_dummies(user_features.set_index("userId"), columns=["region", "gender"])

        # 2. 유저 정보를 df에 merge
        df_merged = df.drop(columns=["orderType"]).merge(user_features_encoded, on="userId")

        # 3. 문자열 컬럼 제거 (평균 계산에 필요한 수치형 컬럼만 남기기)
        df_numeric = df_merged.drop(columns=["region","gender","product", "userId", "sellerId", "productCategory", "timestamp"])

        # 4. 제품별 사용자 특성 평균
        product_user_features = df_numeric.groupby(df_merged["product"]).mean()

        # 5. 표준화
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(product_user_features)

        # 6. 유사도 계산
        similarity = cosine_similarity(scaled_features)
        similarity_df = pd.DataFrame(similarity, index=product_user_features.index, columns=product_user_features.index)

        model = {
            "similarity_df": similarity_df,
            "feature_columns": product_user_features,
            "scaled_features": scaled_features,
            "product_names": product_user_features.index.tolist()  # product 이름 리스트
        }
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        return {"message": "Content-based filtering model trained and saved."}

    elif algo_name == "collaborative":
        user_item_matrix = df.groupby(["userId", "product"]).size().unstack(fill_value=0)
        similarity = cosine_similarity(user_item_matrix)
        similarity_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

        with open(model_path, "wb") as f:
            pickle.dump(similarity_df, f)

        return {"message": "Collaborative filtering model trained and saved."}

    elif algo_name == "svd":
        user_item_matrix = df.groupby(["userId", "product"]).size().unstack(fill_value=0)
        svd = TruncatedSVD(n_components=10)
        svd_matrix = svd.fit_transform(user_item_matrix)

        model_data = {
            "svd": svd,
            "user_index": user_item_matrix.index.tolist(),
            "item_columns": user_item_matrix.columns.tolist()
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        return {"message": "SVD recommendation model trained and saved."}

    elif algo_name == "xgb_classifier":
        # 구매된 조합 (positive sample)
        df["label"] = 1
        positive_df = df[["userId", "product", "age", "gender", "region", "label"]]

        # 사용자 및 상품 목록
        users = df["userId"].unique()
        products = df["product"].unique()

        # 음성 샘플 생성 (랜덤 사용자-상품 조합 중 실제 구매한 것 제외)
        import random

        neg_samples = []
        existing_pairs = set(zip(df["userId"], df["product"]))

        while len(neg_samples) < len(positive_df) * 0.5:
            user = random.choice(users)
            product = random.choice(products)
            if (user, product) not in existing_pairs:
                user_info = user_features.loc[user]
                neg_samples.append({
                    "userId": user,
                    "product": product,
                    "age": user_info["age"],
                    "gender": user_info["gender"],
                    "region": user_info["region"],
                    "label": 0
                })

        negative_df = pd.DataFrame(neg_samples)
        train_df = pd.concat([positive_df, negative_df], ignore_index=True)

        product_encoder = {k: v for v, k in enumerate(df["product"].astype("category").cat.categories)}
        region_encoder = {k: v for v, k in enumerate(df["region"].astype("category").cat.categories)}

        X = pd.DataFrame({
            "age": train_df["age"],
            "gender": train_df["gender"].map({"남": 0, "여": 1}),
            "region": train_df["region"].map(region_encoder),
            "product": train_df["product"].map(product_encoder)
        })
        y = train_df["label"]

        model = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.2,
                              use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)

        model_data = {
            "model": model,
            "product_encoder": product_encoder,
            "region_encoder": region_encoder,
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        return {"message": "XGBoost classifier recommendation model trained and saved."}


    elif algo_name == "knn":
        user_item_matrix = df.groupby(["userId", "product"]).size().unstack(fill_value=0)
        knn = NearestNeighbors(n_neighbors=5, metric="cosine")
        knn.fit(user_item_matrix)

        model_data = {
            "knn": knn,
            "user_index": user_item_matrix.index.tolist(),
            "product_columns": user_item_matrix.columns.tolist(),
            "user_item_matrix": user_item_matrix.values  # 사용자 벡터 저장
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        return {"message": "k-NN recommendation model trained and saved."}

    elif algo_name == "item2vec":
        # 유저별 구매 시퀀스 생성
        user_sequence = df.groupby("userId")["product"].apply(list).tolist()

        # Word2Vec 훈련 (Skip-gram 방식)
        model = Word2Vec(sentences=user_sequence, vector_size=50, window=5, min_count=1, sg=1, workers=4)

        model_data = {
            "item2vec_model": model,
            "product_list": list(model.wv.index_to_key)
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        return {"message": "Item2Vec model trained and saved."}

    else:
        raise ValueError(f"Unsupported recommendation model: {algo_name}")
