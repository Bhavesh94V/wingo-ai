import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from supabase import create_client, Client
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Wingo AI Predictor", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://npbpjsdxisdutcruwkgr.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5wYnBqc2R4aXNkdXRjcnV3a2dyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzk0MzMxOTEsImV4cCI6MjA5NTAwOTE5MX0.Td38AsexT9B7C6LSuFBml3QVFaaMn-m-rcJgXtQ_uIU")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class RoundData(BaseModel):
    number: int
    colour: str
    big_small: str

class PredictRequest(BaseModel):
    game_code: str
    period: Optional[str] = ""
    recent_rounds: List[RoundData]

class PredictResponse(BaseModel):
    period: str
    prediction: str           # "BIG" or "SMALL"
    confidence: float         # 0-100
    colour_prediction: str    # "Red", "Green", "Violet", "Red/Violet"
    top_numbers: List[int]    # Top 3 predicted numbers
    model_used: str
    features_used: int
    reasoning: str

def fetch_history(game_code: str, limit: int = 600):
    try:
        res = supabase.table("rounds")\
            .select("*")\
            .eq("game_code", game_code)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        return res.data
    except Exception as e:
        print(f"Supabase fetch error: {e}")
        return []

def get_colour(num: int) -> str:
    if num == 0: return "red,violet"
    if num == 5: return "green,violet"
    if num in [1,3,7,9]: return "green"
    return "red"

def build_features(rows: list):
    if len(rows) < 15:
        return pd.DataFrame(), []

    df = pd.DataFrame(rows)
    df = df.sort_values("created_at").reset_index(drop=True)
    df['is_big']    = (df['big_small'] == 'Big').astype(int)
    df['num']       = df['number'].astype(int)
    df['is_red']    = df['colour'].str.contains('red').astype(int)
    df['is_green']  = df['colour'].str.contains('green').astype(int)
    df['is_violet'] = df['colour'].str.contains('violet').astype(int)

    features, labels, num_labels = [], [], []

    for i in range(10, len(df) - 1):
        window = df.iloc[i-10:i]
        target = df.iloc[i+1]
        last_10_big = list(window['is_big'])

        streak = 0
        last_val = last_10_big[-1]
        for v in reversed(last_10_big):
            if v == last_val: streak += 1
            else: break

        big_ratio   = sum(last_10_big) / 10
        last5_ratio = sum(last_10_big[-5:]) / 5
        alternating = sum(1 for j in range(1, 10) if last_10_big[j] != last_10_big[j-1]) / 9
        last_nums   = list(window['num'].tail(3))
        avg_num     = window['num'].mean()
        std_num     = window['num'].std()
        p3          = last_10_big[-3:]
        ngram_3     = p3[0]*4 + p3[1]*2 + p3[2]
        violet_cnt  = window['is_violet'].tail(5).sum()
        zero_cnt    = (window['num'].tail(5) == 0).sum()
        mean_dev    = avg_num - 4.5
        # Colour counts last 5
        red_cnt     = window['is_red'].tail(5).sum()
        green_cnt   = window['is_green'].tail(5).sum()
        # Last num parity
        last_odd    = int(last_nums[-1] % 2 != 0)

        feat = [
            streak, big_ratio, last5_ratio, alternating,
            *last_10_big,
            ngram_3, violet_cnt, zero_cnt,
            avg_num, std_num, mean_dev,
            last_nums[0], last_nums[1], last_nums[2],
            red_cnt, green_cnt, last_odd
        ]
        features.append(feat)
        labels.append(int(target['is_big']))
        num_labels.append(int(target['num']))

    return pd.DataFrame(features), labels, num_labels

def build_one_feature(rounds_data: list):
    """Build a single feature row from last 11 rounds"""
    df = pd.DataFrame(rounds_data)
    df['is_big']    = (df['big_small'] == 'Big').astype(int)
    df['is_red']    = df['colour'].str.contains('red', case=False).astype(int)
    df['is_green']  = df['colour'].str.contains('green', case=False).astype(int)
    df['is_violet'] = df['colour'].str.contains('violet', case=False).astype(int)
    df['num']       = df['number'].astype(int)

    window = df.head(10)
    last_10_big = list(window['is_big'])
    streak = 0
    last_val = last_10_big[-1]
    for v in reversed(last_10_big):
        if v == last_val: streak += 1
        else: break

    big_ratio   = sum(last_10_big) / 10
    last5_ratio = sum(last_10_big[-5:]) / 5
    alternating = sum(1 for j in range(1, 10) if last_10_big[j] != last_10_big[j-1]) / 9
    last_nums   = list(window['num'].tail(3))
    avg_num     = window['num'].mean()
    std_num     = window['num'].std()
    p3          = last_10_big[-3:]
    ngram_3     = p3[0]*4 + p3[1]*2 + p3[2]
    violet_cnt  = window['is_violet'].tail(5).sum()
    zero_cnt    = (window['num'].tail(5) == 0).sum()
    mean_dev    = avg_num - 4.5
    red_cnt     = window['is_red'].tail(5).sum()
    green_cnt   = window['is_green'].tail(5).sum()
    last_odd    = int(last_nums[-1] % 2 != 0)

    return np.array([[
        streak, big_ratio, last5_ratio, alternating,
        *last_10_big,
        ngram_3, violet_cnt, zero_cnt,
        avg_num, std_num, mean_dev,
        last_nums[0], last_nums[1], last_nums[2],
        red_cnt, green_cnt, last_odd
    ]]), streak, last_val, big_ratio

def predict_top_numbers(big_small: str, num_model, x_pred, num_labels) -> List[int]:
    """Predict top 3 likely numbers using number model"""
    try:
        num_probs = num_model.predict_proba(x_pred)[0]
        classes   = num_model.classes_
        # Filter by big/small
        filtered = [(classes[i], num_probs[i]) for i in range(len(classes))
                    if (big_small == "BIG" and classes[i] >= 5) or
                       (big_small == "SMALL" and classes[i] <= 4)]
        if not filtered:
            filtered = [(classes[i], num_probs[i]) for i in range(len(classes))]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [int(n) for n, _ in filtered[:3]]
    except:
        return [5, 7, 6] if big_small == "BIG" else [1, 3, 4]

@app.get("/")
def root():
    return {"status": "Wingo AI Server Running ✅", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    game_code = req.game_code
    period    = req.period or ""

    history = fetch_history(game_code, limit=600)

    if len(history) < 30:
        history = [
            {
                "number":     r.number,
                "colour":     r.colour,
                "big_small":  r.big_small,
                "created_at": f"2024-01-01T00:00:{str(i).zfill(2)}Z",
                "game_code":  game_code
            }
            for i, r in enumerate(reversed(req.recent_rounds))
        ]

    result = build_features(history)

    if isinstance(result[0], pd.DataFrame) and result[0].empty:
        return PredictResponse(
            period=period, prediction="BIG", confidence=50.0,
            colour_prediction="Red", top_numbers=[5,7,6],
            model_used="fallback", features_used=0,
            reasoning="Not enough data. Need 50+ rounds."
        )

    X, y, y_num = result

    if len(X) < 20:
        return PredictResponse(
            period=period, prediction="BIG", confidence=50.0,
            colour_prediction="Red", top_numbers=[5,7,6],
            model_used="fallback", features_used=0,
            reasoning=f"Only {len(X)} samples. Playing safe."
        )

    # Train Big/Small model
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=42)
    rf.fit(X, y)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    gb.fit(X, y)

    # Train Number model
    num_rf = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=2, random_state=42)
    try:
        num_rf.fit(X, y_num)
    except:
        num_rf = None

    # Build feature for current prediction
    recent_data = [
        {"number": r.number, "colour": r.colour, "big_small": r.big_small}
        for r in req.recent_rounds[:11]
    ]
    x_pred, streak, last_val, big_ratio = build_one_feature(recent_data)

    # Ensemble: RF 60% + GB 40%
    rf_prob  = rf.predict_proba(x_pred)[0][1]
    gb_prob  = gb.predict_proba(x_pred)[0][1]
    ensemble = 0.6 * rf_prob + 0.4 * gb_prob

    prediction = "BIG" if ensemble >= 0.5 else "SMALL"
    confidence = round(max(ensemble, 1 - ensemble) * 100, 1)

    # Colour prediction
    if prediction == "BIG":
        colour_pred = "Red"       # 6,8,9 → red | 7 → green
        if rf_prob < 0.55: colour_pred = "Red/Green"
    else:
        colour_pred = "Green"     # 1,3 → green | 2,4 → red
        if big_ratio > 0.45: colour_pred = "Red/Green"

    # Top numbers
    top_nums = predict_top_numbers(prediction, num_rf, x_pred, y_num) if num_rf else (
        [5,7,6] if prediction == "BIG" else [1,3,4]
    )

    reasoning = (
        f"RF={round(rf_prob*100)}% BIG | "
        f"GB={round(gb_prob*100)}% BIG | "
        f"Streak={streak} {'BIG' if last_val else 'SMALL'} | "
        f"Big%={round(big_ratio*100)}% | "
        f"Samples={len(X)}"
    )

    return PredictResponse(
        period=period,
        prediction=prediction,
        confidence=confidence,
        colour_prediction=colour_pred,
        top_numbers=top_nums,
        model_used=f"RF+GB+Num ({len(X)} samples)",
        features_used=len(x_pred[0]),
        reasoning=reasoning
    )
