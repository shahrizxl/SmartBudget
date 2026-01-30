import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ======================================================
# TIME SERIES FEATURE ENGINEERING
# ======================================================

def aggregate_expenses(df):
    """
    Aggregates transactions into daily total expenses and creates features.
    Returns: (series_df, None) or (empty_df, "No data.")
    """

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

    if len(series_df) < 30:
        return pd.DataFrame(), "No data."

    return series_df, None


# ======================================================
# MODEL TRAINING
# ======================================================

def train_random_forest(series_df):
    features = [c for c in series_df.columns if c != "daily_expense"]
    X = series_df[features]
    y = series_df["daily_expense"]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)
    return model, features


# ======================================================
# MODEL EVALUATION
# ======================================================

def calculate_accuracy_metrics(model, X_test, y_test, fallback_tol=1.2):
    if X_test is None or X_test.empty:
        return None, "Fallback average."

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    baseline = np.full_like(y_test, y_test.mean(), dtype=float)
    baseline_mae = mean_absolute_error(y_test, baseline)

    accepted = (
        (baseline_mae == 0 and mae == 0) or
        (mae <= baseline_mae * fallback_tol) or
        (r2 > 0)
    )

    if not accepted:
        return None, "Fallback average."

    return {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 3)
    }, None


# ======================================================
# FUTURE PREDICTION (NaN-SAFE)
# ======================================================

def predict_next_days(model, model_features, series_df, steps=30):
    if series_df is None or series_df.empty:
        return pd.DataFrame(), None, "No data."

    last_date = series_df.index[-1]
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=steps)

    temp_df = series_df.copy()
    preds = []

    for d in future_dates:
        tail = temp_df["daily_expense"].tail(7)
        rm3 = tail.tail(3)

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
            "rolling_mean_7": tail.mean() if not tail.empty else 0.0,
            "rolling_std_7": tail.std() if tail.std() == tail.std() else 0.0,
            "cumsum_7": tail.sum()
        }

        X_pred = pd.DataFrame([row])
        for f in model_features:
            if f not in X_pred:
                X_pred[f] = 0.0
        X_pred = X_pred[model_features]

        pred = max(0.0, float(model.predict(X_pred)[0]))

        preds.append(pred)
        temp_df.loc[d] = pred

    predictions_df = pd.DataFrame(
        {"daily_expense_pred": preds},
        index=future_dates
    )

    n_test = min(14, len(series_df))
    if n_test > 0:
        test_df = series_df.tail(n_test)
        metrics, _ = calculate_accuracy_metrics(
            model,
            test_df[model_features],
            test_df["daily_expense"]
        )
        return predictions_df, metrics, f"Random Forest backtested on {n_test} days"

    return predictions_df, None, "No backtest data."


# ======================================================
# MASTER FUNCTION (FLASK-SAFE)
# ======================================================

def predict_all_horizon(transactions_df):
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

    model, features = train_random_forest(series_df)
    preds_df, metrics, _ = predict_next_days(model, features, series_df)

    if preds_df.empty:
        return (
            "Prediction failed. Using averages.",
            fallback_day,
            fallback_week,
            fallback_month,
            None
        )

    next_day = round(float(preds_df.iloc[0]["daily_expense_pred"]), 2)
    next_week = round(float(preds_df.head(7)["daily_expense_pred"].sum()), 2)
    next_month = round(float(preds_df.head(30)["daily_expense_pred"].sum()), 2)

    if metrics:
        return (
            "Machine Learning model used.",
            next_day,
            next_week,
            next_month,
            metrics
        )

    return (
        "Prediction complete using fallback averages.",
        fallback_day,
        fallback_week,
        fallback_month,
        None
    )
