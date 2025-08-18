import pickle
import os
import pandas as pd
from dateutil.relativedelta import relativedelta

model_dir = "/app/model_storage/predict"

def predict_quantity_pipeline(product_name: str, algo: str):
    model_path = os.path.join(model_dir, f"model_{algo}.pkl")
    print(algo)
    if not os.path.exists(model_path):
        return {"error": f"Model file not found for algorithm: {algo}"}

    with open(model_path, "rb") as f:
        models = pickle.load(f)

    if algo in ["prophet", "xgb_timeseries", "arima","sarimax"]:
        if product_name not in models:
            return {"error": "No model for given product"}

        model_info = models[product_name]
        preds = model_info["predictions"]
        last_date = model_info.get("last_date")

        if last_date:
            if not isinstance(last_date, pd.Timestamp):
                last_date = pd.to_datetime(last_date)
            future_months = [
                (last_date + relativedelta(months=i)).strftime("%Y-%m") for i in range(1,5)
            ]
        else:
            future_months = [f"Month+{i}"for i in range(1,5)]

        return {
            "product" : product_name,
            "algorithm" : algo,
            "predictions" : preds,
            "future_months" : future_months
        }

    else:
        # 기존 파이프라인 방식 (기존 회귀모델)
        pipeline, df_grouped = models
        row = df_grouped[df_grouped["productName"] == product_name]
        if row.empty:
            return {"error": "Product not found"}
        X_pred = row.drop(columns=["productName", "productQuantity"])
        preds = pipeline.predict(X_pred).tolist()

        return {
            "product": product_name,
            "algorithm": algo,
            "predictions": preds
        }