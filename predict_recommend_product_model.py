import pickle
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_dir = "/app/model_storage/recommend"

def encode_user_info(user_info, feature_columns):
    # 입력된 유저 정보를 수치형으로 인코딩
    user_df = pd.DataFrame([user_info])

    # 범주형 수치화
    user_encoded = pd.get_dummies(user_df, columns=["region", "gender"])

    # 누락된 더미 컬럼 추가 (product_user_features 기준과 일치시키기 위해)
    for col in feature_columns:
        if col not in user_encoded.columns:
            user_encoded[col] = 0

    # 컬럼 순서 정렬 (동일하게 맞춰줘야 함)
    user_encoded = user_encoded[feature_columns]

    return user_encoded

def predict_recommendation_pipeline(user_info: dict, algo: str):
    model_path = os.path.join(model_dir, f"model_{algo}.pkl")

    if not os.path.exists(model_path):
        return {"error": f"Model file not found for algorithm: {algo}"}

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    if algo == "content":
        # 콘텐츠 기반: 사용자와 유사한 product feature 찾기
        similarity_df = model_data["similarity_df"]
        feature_columns = model_data["feature_columns"]
        # 사용자 벡터 생성
        user_vector = encode_user_info(user_info, feature_columns.columns).values.reshape(1, -1)
        # 유사도 계산 (1xN)
        product_vectors = model_data["scaled_features"]
        product_names = model_data["product_names"]

        cos_scores = cosine_similarity(user_vector, product_vectors).flatten()  # 유사도 점수 (1차원)
        top_n_idx = cos_scores.argsort()[::-1][:10]  # 높은 순서 Top 5

        top_products = [product_names[i] for i in top_n_idx]
        return {
            "algorithm": algo,
            "recommended_products": top_products
        }

    elif algo == "collaborative":
        similarity_df = model_data
        # 단순히 가장 유사도가 높은 사용자 5명 → 그들이 산 제품을 추천
        user_id = str(user_info.get("userId", ""))
        if user_id not in similarity_df.index:
            return {"error": "User not found in collaborative matrix"}

        similar_users = similarity_df.loc[user_id].sort_values(ascending=False).head(5).index.tolist()
        # 기존 구매 로그에서 비슷한 사용자들이 많이 구매한 상품을 추천
        # ※ 실제 운영 시에는 구매 로그 전체에서 유사 사용자 상품 수집 필요
        return {
            "algorithm": algo,
            "recommended_users": similar_users,
            "note": "추가적으로 이들의 구매 로그를 활용해 제품을 추출해야 합니다."
        }

    elif algo == "svd":
        svd_model = model_data["svd"]
        user_index = model_data["user_index"]
        item_columns = model_data["item_columns"]

        # 새 사용자라면 zero 벡터 또는 평균으로 대체
        user_vector = np.zeros((1, len(item_columns)))
        svd_user_proj = svd_model.transform(user_vector)
        scores = svd_model.inverse_transform(svd_user_proj).flatten()
        top_indices = scores.argsort()[-5:][::-1]
        top_products = [item_columns[i] for i in top_indices]

        return {
            "algorithm": algo,
            "recommended_products": top_products
        }

    elif algo == "xgb_classifier":
        model = model_data["model"]
        product_encoder = model_data["product_encoder"]
        region_encoder = model_data["region_encoder"]

        age = user_info.get("age", 30)
        gender = 0 if user_info.get("gender") == "M" else 1
        region = region_encoder.get(user_info.get("region", ""), 0)

        X_input = []
        product_map = {}
        for product, code in product_encoder.items():
            X_input.append([age, gender, region, code])
            product_map[code] = product

        preds = model.predict_proba(np.array(X_input))[:, 1]
        top_indices = preds.argsort()[-5:][::-1]
        top_products = [product_map[X_input[i][3]] for i in top_indices]

        return {
            "algorithm": algo,
            "recommended_products": top_products
        }
    elif algo == "knn":
        knn_model = model_data["knn"]
        user_index = model_data["user_index"]
        product_columns = model_data["product_columns"]
        user_matrix = model_data["user_item_matrix"]

        userId = str(user_info.get("userId", ""))
        if userId not in user_index:
            return {"error": f"User '{userId}' not found in training data"}

        user_idx = user_index.index(userId)
        user_vector = user_matrix[user_idx].reshape(1, -1)

        distances, indices = knn_model.kneighbors(user_vector)
        neighbor_indices = indices.flatten()
        neighbor_userIds = [user_index[i] for i in neighbor_indices if i < len(user_index)]

        from collections import Counter
        neighbor_purchases = []
        for neighbor in neighbor_userIds:
            neighbor_vector = user_matrix[user_index.index(neighbor)]
            purchased = [product_columns[i] for i, val in enumerate(neighbor_vector) if val > 0]
            neighbor_purchases.extend(purchased)

        top_products = [p for p, _ in Counter(neighbor_purchases).most_common(5)]
        return {"algorithm": algo, "recommended_products": top_products}

    elif algo == "item2vec":
        item2vec_model = model_data["item2vec_model"]
        product_list = model_data["product_list"]

        # 해당 유저의 구매 이력 추출
        user_products = df[df["userId"] == user_id]["product"].tolist()

        if not user_products:
            return {"message": "No purchase history for user."}

        # 최근에 본 상품 하나 기준으로 유사한 상품 추천 (여러 개면 평균 벡터도 가능)
        last_product = user_products[-1]

        if last_product not in product_list:
            return {"message": f"Product '{last_product}' not in trained model."}

        # 유사한 상품 추천
        similar_items = model.wv.most_similar(last_product, topn=topn)

        return {
            "user_id": user_id,
            "base_product": last_product,
            "recommendations": [item for item, score in similar_items]
        }

    else:
        return {"error": f"Unsupported algorithm: {algo}"}
