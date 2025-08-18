import pandas as pd
import pickle
import os
from elasticsearch import Elasticsearch
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from config import Config
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix

model_dir = "/app/model_storage/predict"
os.makedirs(model_dir, exist_ok=True)

def fetch_all_es_data(index_name, es, scroll='2m', size=1000):
    all_data = []
    page = es.search(
        index=index_name,
        scroll=scroll,
        size=size,
        body={"query": {"match_all": {}}}
    )
    sid = page['_scroll_id']
    hits = page['hits']['hits']
    all_data.extend(hits)

    while len(hits) > 0:
        page = es.scroll(scroll_id=sid, scroll=scroll)
        sid = page['_scroll_id']
        hits = page['hits']['hits']
        all_data.extend(hits)

    return [doc['_source'] for doc in all_data]

def train_predict_model_and_save(algo_name: str):
    # if algo_name in ["linear", "logistic", "tree", "xgb"]:
    #     return train_standard_model(algo_name)
    # el
    timeseries_algos = ["xgb_timeseries", "prophet", "arima", "sarimax"]
    if algo_name in timeseries_algos:
        return train_timeseries_model(algo_name)
    else:
        raise ValueError(f"Unsupported or invalid algorithm: {algo_name}")

def train_timeseries_model(algo_name: str):

    es = Elasticsearch(Config.ELASTICSEARCH_URI)
    index_name = "order_products-logs"
    data = fetch_all_es_data(index_name, es)
    df = pd.DataFrame(data)

    # 날짜 변환 및 필터링
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year_month"] = df["timestamp"].dt.to_period("M")

    # 월별 판매량 집계
    monthly_sales = df.groupby(["productName", "year_month"])["productQuantity"].sum().reset_index()
    monthly_sales["year_month"] = monthly_sales["year_month"].astype(str)
    monthly_sales["year_month"] = pd.to_datetime(monthly_sales["year_month"])
    monthly_sales = monthly_sales.sort_values(["productName", "year_month"])

    models = {}

    for product in monthly_sales["productName"].unique():
        df_product = monthly_sales[monthly_sales["productName"] == product].copy()
        # 누락된 월 채우기 (보간)
        all_months = pd.date_range(
            start=monthly_sales["year_month"].min(),
            end=monthly_sales["year_month"].max(),
            freq="MS"
        )
        # 날짜를 index로 바꿔서 시계열 처럼 다루기 쉽도록 함
        df_product = df_product.set_index("year_month").reindex(all_months)
        df_product.index.name = "year_month"
        df_product["productName"] = product
        # 누락 되었던 월의 productQuantity 부분의 데이터 선형 보간
        df_product["productQuantity"] = df_product["productQuantity"].astype(float).interpolate(method="linear")
        # index로 변했던 부분을 다시 일반 컬럼으로 되돌림
        df_product = df_product.reset_index()

        if len(df_product) < 1:
            continue  # 데이터가 너무 적으면 스킵

        if algo_name == "xgb_timeseries":
            # lag feature 생성
            n_lags = 3
            for lag in range(1, n_lags + 1):
                df_product[f"lag_{lag}"] = df_product["productQuantity"].shift(lag)
            df_product["target"] = df_product["productQuantity"].shift(-1)
            df_product.dropna(inplace=True)

            X = df_product[["lag_1", "lag_2", "lag_3"]]
            y = df_product["target"]
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, objective='reg:squarederror')
            model.fit(X, y)

            last_date = df_product["year_month"].max()

            # 향후 4개월 예측
            future_preds = []
            last_vals = list(df_product.iloc[-3:]["productQuantity"])
            for _ in range(4):
                x_input = pd.DataFrame([last_vals[-3:]], columns=["lag_1", "lag_2", "lag_3"])
                pred = model.predict(x_input)[0]
                future_preds.append(pred)
                last_vals.append(pred)

            models[product] = {
                "model": model,
                "last_vals": df_product.iloc[-3:]["productQuantity"].tolist(),
                "predictions": future_preds,
                "last_date" : last_date
            }

        elif algo_name == "prophet":
            df_prophet = df_product.rename(columns={"year_month": "ds", "productQuantity": "y"})
            model = Prophet()
            model.fit(df_prophet)

            last_date = df_prophet["ds"].max()
            future = pd.date_range(start=last_date + relativedelta(months=1), periods=4, freq="MS")
            future_df = pd.DataFrame({"ds": future})

            forecast = model.predict(future_df)
            preds = forecast["yhat"].tolist()

            models[product] = {
                "model": model,
                "predictions": preds,
                "last_date" : last_date
            }
        elif algo_name == "arima":
            try:
                model = ARIMA(df_product["productQuantity"], order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=4)
                last_date = df_product["year_month"].max()

                models[product] = {
                    "model": model_fit,
                    "predictions": forecast.tolist(),
                    "last_date" : last_date
                }
            except Exception as e:
                print(f"ARIMA failed for {product}: {e}")
                continue

        elif algo_name == "sarimax":
            try:
                model = SARIMAX(df_product["productQuantity"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(steps=4)
                last_date = df_product["year_month"].max()
                models[product] = {
                    "model": model_fit,
                    "predictions": forecast.tolist(),
                    "last_date": last_date
                }
            except Exception as e:
                print(f"SARIMAX failed for {product}: {e}")
                continue

    # 🔹 저장
    model_path = os.path.join(model_dir, f"model_{algo_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(models, f)

    print(models.keys())
    return {"message": f"{algo_name} time series model trained and saved."}

# def get_model_by_name(algo_name: str):
#     if algo_name == "linear":
#         return LinearRegression()
#     elif algo_name == "logistic":
#         return LogisticRegression()
#     elif algo_name == "tree":
#         return DecisionTreeRegressor()
#     elif algo_name == "xgb":
#         return XGBRegressor()
#     else:
#         raise ValueError(f"Unsupported algorithm: {algo_name}")

# def train_standard_model(algo_name: str):
#     es = Elasticsearch("http://localhost:9200")
#     index_name = "order_products-logs"
#
#     res = es.search(index=index_name, body={"size": 10000, "query": {"match_all": {}}})
#     data = [hit['_source'] for hit in res['hits']['hits']]
#     df = pd.DataFrame(data)
#
#     # 🔹 날짜 전처리
#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     df["month"] = df["timestamp"].dt.month
#     df["year_month"] = df["timestamp"].dt.to_period("M")
#
#     # 🔹 성별 전처리 (남: 0, 여: 1)
#     df["userGender"] = df["userGender"].map({"남": 0, "여": 1})
#
#     # 🔹 전월 판매량 계산 → 증가율 (product 기준)
#     monthly_sales = df.groupby(["productName", "year_month"])["productQuantity"].sum().reset_index()
#     monthly_sales["prev_month_qty"] = monthly_sales.groupby("productName")["productQuantity"].shift(1)
#     monthly_sales["increase_rate"] = (
#         (monthly_sales["productQuantity"] - monthly_sales["prev_month_qty"]) /
#         monthly_sales["prev_month_qty"]
#     ).fillna(0)
#
#     # 🔹 원본 df와 병합
#     df["year_month"] = df["timestamp"].dt.to_period("M")
#     df = pd.merge(df, monthly_sales[["productName", "year_month", "increase_rate"]],
#                   on=["productName", "year_month"], how="left")
#
#     df["increase_rate"] = df["increase_rate"].fillna(0)
#
#     # 🔹 필요한 열만 추출
#     features = ["userAge", "productPrice", "userGender", "userRegion", "month", "increase_rate"]
#     df_model = df[features + ["productName", "productQuantity"]]
#
#     # 🔹 학습용 데이터 준비: 제품별 평균값 사용
#     df_grouped = df_model.groupby("productName").agg({
#         "userAge": "mean",
#         "productPrice": "mean",
#         "userGender": "mean",
#         "month": "mean",
#         "increase_rate": "mean",
#         "userRegion": lambda x: x.mode()[0],  # 최빈값 사용
#         "productQuantity": "sum"
#     }).reset_index()
#
#     # 🔹 Feature / Target 구분
#     X = df_grouped[["userAge", "productPrice", "userGender", "userRegion", "month", "increase_rate"]]
#     y = df_grouped["productQuantity"]
#
#     # 🔹 범주형 인코딩 (userRegion)
#     categorical_features = ["userRegion"]
#     numeric_features = ["userAge", "productPrice", "userGender", "month", "increase_rate"]
#
#     preprocessor = ColumnTransformer(transformers=[
#         ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
#     ], remainder="passthrough")
#
#     model = get_model_by_name(algo_name)
#
#     # 🔹 파이프라인 구성
#     pipeline = Pipeline(steps=[
#         ("preprocessor", preprocessor),
#         ("model", model)
#     ])
#
#     pipeline.fit(X, y)
#
#     # 🔹 저장
#     model_path = os.path.join(model_dir, f"model_{algo_name}.pkl")
#     with open(model_path, "wb") as f:
#         pickle.dump((pipeline, df_grouped), f)
#
#     return {"message": f"{algo_name} model trained and saved."}
