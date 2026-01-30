import pandas as pd
import numpy as np
from datetime import timedelta

from sklearn.ensemble import (
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ======================================================
# TIME SERIES FEATURE ENGINEERING
# ======================================================

def aggregate_expenses(df):
    if df is None or df.empty:
        return pd.DataFrame(), "No data."

    df = df.copy()

    if "type" in df.columns:
        df = df[df["type"] == "expense"]

    if df.empty or "date" not in df.columns or "amount" not in df.columns:
        return pd.DataFrame(), "No data."

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df.dropna(subset=["date", "amount"], inplace=True)

    if df.empty:
        return pd.DataFrame(), "No data."

    daily = df.groupby("date")["amount"].sum()

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    series_df = daily.reindex(full_range, fill_value=0).to_frame("daily_expense")

    series_df["day_of_week"] = series_df.index.dayofweek
    series_df["is_weekend"] = series_df["day_of_week"].isin([5, 6]).astype(int)
    series_df["month"] = series_df.index.month
    series_df["day"] = series_df.index.day

    series_df["lag_1"] = series_df["daily_expense"].shift(1)
    series_df["lag_2"] = series_df["daily_expense"].shift(2)
    series_df["lag_3"] = series_df["daily_expense"].shift(3)

    series_df["rolling_mean_3"] = series_df["daily_expense"].rolling(3).mean().shift(1)
    series_df["rolling_std_3"] = (
        series_df["daily_expense"].rolling(3).std().shift(1).fillna(0)
    )

    series_df["rolling_mean_7"] = series_df["daily_expense"].rolling(7).mean().shift(1)
    series_df["rolling_std_7"] = (
        series_df["daily_expense"].rolling(7).std().shift(1).fillna(0)
    )

    series_df["cumsum_7"] = series_df["daily_expense"].rolling(7).sum().shift(1)

    series_df.dropna(inplace=True)

    if len(series_df) < 10:
        return pd.DataFrame(), "No data."

    return series_df, None


# ======================================================
# TRAIN MULTIPLE MODELS
# ======================================================

def train_all_models(series_df):
    X = series_df.drop(columns=["daily_expense"])
    y = series_df["daily_expense"]

    models = {
        "Ridge": Ridge(alpha=1.0),
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )
        
    }

    trained = {}
    for name, model in models.items():
        model.fit(X, y)
        trained[name] = model

    return trained, X.columns.tolist()


# ======================================================
# MODEL EVALUATION
# ======================================================

def evaluate_models(models, X_test, y_test, fallback_tol=1.2):
    baseline = np.full_like(y_test, y_test.mean(), dtype=float)
    baseline_mae = mean_absolute_error(y_test, baseline)

    results = []

    for name, model in models.items():
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        accepted = (
            mae <= baseline_mae * fallback_tol or
            r2 > 0
        )

        results.append({
            "model": name,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 3),
            "accepted": accepted
        })

    return pd.DataFrame(results).sort_values("mae")


# ======================================================
# FUTURE PREDICTION (NO NaNs, FEATURE-SAFE)
# ======================================================

def predict_future(model, features, series_df, days=30):
    last_date = series_df.index[-1]
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=days)

    temp_df = series_df.copy()
    predictions = []

    for d in future_dates:
        tail = temp_df["daily_expense"].tail(7)
        rm3 = tail.tail(3)
        rm7 = tail

        row = {
            "day_of_week": d.dayofweek,
            "is_weekend": int(d.dayofweek in [5, 6]),
            "month": d.month,
            "day": d.day,
            "lag_1": tail.iloc[-1] if len(tail) >= 1 else 0.0,
            "lag_2": tail.iloc[-2] if len(tail) >= 2 else 0.0,
            "lag_3": tail.iloc[-3] if len(tail) >= 3 else 0.0,
            "rolling_mean_3": rm3.mean() if not rm3.empty else 0.0,
            "rolling_std_3": rm3.std() if rm3.std() == rm3.std() else 0.0,
            "rolling_mean_7": rm7.mean() if not rm7.empty else 0.0,
            "rolling_std_7": rm7.std() if rm7.std() == rm7.std() else 0.0,
            "cumsum_7": rm7.sum()
        }

        X_pred = pd.DataFrame([row])
        for f in features:
            if f not in X_pred:
                X_pred[f] = 0.0
        X_pred = X_pred[features]

        pred = max(0.0, float(model.predict(X_pred)[0]))

        predictions.append(pred)
        temp_df.loc[d] = pred

    return pd.DataFrame({"daily_expense_pred": predictions}, index=future_dates)


# ======================================================
# MASTER FUNCTION (FLASK-SAFE)
# ======================================================

def predict_all_horizons_multi(transactions_df):
    if transactions_df is None or transactions_df.empty:
        return "No expense data available.", 0.0, 0.0, 0.0, None

    df = transactions_df.copy()
    if "type" in df.columns:
        df = df[df["type"] == "expense"]

    if df.empty or df["amount"].sum() == 0:
        return "No expense data available.", 0.0, 0.0, 0.0, None

    daily = df.groupby("date")["amount"].sum().sort_index()

    fallback_day = round(float(daily.iloc[-1]), 2)
    fallback_week = round(float(daily.tail(7).mean() * 7), 2)
    fallback_month = round(float(daily.tail(30).mean() * 30), 2)

    series_df, msg = aggregate_expenses(transactions_df)
    if msg:
        return (
            "Insufficient data. Using averages.",
            fallback_day,
            fallback_week,
            fallback_month,
            None
        )

    models, features = train_all_models(series_df)

    test_days = min(14, len(series_df))
    test_df = series_df.tail(test_days)

    metrics_df = evaluate_models(
        models,
        test_df[features],
        test_df["daily_expense"]
    )

    accepted = metrics_df[metrics_df["accepted"]]

    if accepted.empty:
        return (
            "ML models not reliable. Using averages.",
            fallback_day,
            fallback_week,
            fallback_month,
            metrics_df.to_dict(orient="records")
        )

    best_model_name = accepted.iloc[0]["model"]
    best_model = models[best_model_name]

    preds_df = predict_future(best_model, features, series_df, days=30)

    next_day = round(float(preds_df.iloc[0]["daily_expense_pred"]), 2)
    next_week = round(float(preds_df.head(7)["daily_expense_pred"].sum()), 2)
    next_month = round(float(preds_df.head(30)["daily_expense_pred"].sum()), 2)

    return (
        f"Predictions generated using {best_model_name}.",
        next_day,
        next_week,
        next_month,
        metrics_df.to_dict(orient="records")
    )
