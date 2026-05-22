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
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Wingo AI Predictor", version="1.0.0")

# Allow all origins (browser script can call this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase setup from environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://npbpjsdxisdutcruwkgr.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5wYnBqc2R4aXNkdXRjcnV3a2dyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzk0MzMxOTEsImV4cCI6MjA5NTAwOTE5MX0.Td38AsexT9B7C6LSuFBml3QVFaaMn-m-rcJgXtQ_uIU")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class RoundData(BaseModel):
    number: int
    colour: str
    big_small: str

class PredictRequest(BaseModel):
    game_code: str
    recent_rounds: List[RoundData]  # Last 30 rounds from browser (newest first)

class PredictResponse(BaseModel):
    prediction: str           # "BIG" or "SMALL"
    confidence: float         # 0-100
    colour_prediction: str    # "Red", "Green", "Violet"
    model_used: str
    features_used: int
    reasoning: str

def fetch_history(game_code: str, limit: int = 500):
    """Fetch historical data from Supabase"""
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

def build_features(rows: list) -> pd.DataFrame:
    """Convert round history into ML features"""
    if len(rows) < 10:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("created_at").reset_index(drop=True)

    # Basic encoding
    df['is_big']    = (df['big_small'] == 'Big').astype(int)
    df['is_red']    = (df['colour'] == 'red').astype(int)
    df['is_green']  = (df['colour'] == 'green').astype(int)
    df['is_violet'] = (df['colour'] == 'violet').astype(int)
    df['num']       = df['number'].astype(int)

    features = []
    labels   = []

    for i in range(10, len(df) - 1):
        window = df.iloc[i-10:i]
        target = df.iloc[i+1]

        # Last 10 outcomes
        last_10_big  = list(window['is_big'])

        # Streak
        streak = 0
        last_val = last_10_big[-1]
        for v in reversed(last_10_big):
            if v == last_val: streak += 1
            else: break

        # Big/Small ratio
        big_ratio = sum(last_10_big) / 10
        last5_ratio = sum(last_10_big[-5:]) / 5

        # Alternating pattern score
        alternating = sum(1 for j in range(1, 10) if last_10_big[j] != last_10_big[j-1]) / 9

        # Last 3 numbers
        last_nums = list(window['num'].tail(3))
        avg_num   = window['num'].mean()
        std_num   = window['num'].std()

        # N-gram: last 3 pattern as binary
        p3 = last_10_big[-3:]
        ngram_3 = p3[0]*4 + p3[1]*2 + p3[2]  # 0-7

        # Violet/0 in last 5
        violet_count = window['is_violet'].tail(5).sum()
        zero_count   = (window['num'].tail(5) == 0).sum()

        # Number deviation from 4.5 mean
        mean_dev = avg_num - 4.5

        feat = [
            streak, big_ratio, last5_ratio, alternating,
            *last_10_big,           # 10 features
            ngram_3, violet_count, zero_count,
            avg_num, std_num, mean_dev,
            last_nums[0], last_nums[1], last_nums[2]
        ]

        features.append(feat)
        labels.append(int(target['is_big']))

    return pd.DataFrame(features), labels

def get_colour_prediction(num_prediction_prob_big: float) -> str:
    """Predict most likely colour based on big/small probability"""
    # Numbers 0: violet/red, 1-4: red/green, 5: violet/green, 6-9: red/green
    if num_prediction_prob_big > 0.65:
        return "Red"   # Big numbers skew red
    elif num_prediction_prob_big < 0.35:
        return "Green" # Small numbers skew green
    else:
        return "Red"   # Default to red (most common)

@app.get("/")
def root():
    return {"status": "Wingo AI Server Running ✅", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    game_code = req.game_code

    # Fetch full history from Supabase
    history = fetch_history(game_code, limit=600)

    if len(history) < 30:
        # Fallback: use only what browser sent
        history = [
            {
                "number": r.number,
                "colour": r.colour,
                "big_small": r.big_small,
                "created_at": f"2024-01-01T00:00:0{i}Z",
                "game_code": game_code
            }
            for i, r in enumerate(reversed(req.recent_rounds))
        ]

    result = build_features(history)

    if isinstance(result, pd.DataFrame) and result.empty:
        return PredictResponse(
            prediction="BIG",
            confidence=50.0,
            colour_prediction="Red",
            model_used="fallback",
            features_used=0,
            reasoning="Not enough data yet. Need at least 50 rounds."
        )

    X, y = result

    if len(X) < 20:
        return PredictResponse(
            prediction="BIG",
            confidence=50.0,
            colour_prediction="Red",
            model_used="fallback",
            features_used=0,
            reasoning=f"Only {len(X)} samples. Need more data."
        )

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42
    )
    rf.fit(X, y)

    # Train Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    gb.fit(X, y)

    # Predict on the LATEST window (last 10 rounds from browser)
    recent = []
    for r in req.recent_rounds:
        recent.append({
            "number": r.number,
            "colour": r.colour,
            "big_small": r.big_small,
            "created_at": "2024-01-01T00:00:00Z",
            "game_code": game_code
        })

    # Use most recent 11 to build 1 feature row
    recent_df = pd.DataFrame(recent[:11])
    recent_df['is_big']    = (recent_df['big_small'] == 'Big').astype(int)
    recent_df['is_red']    = (recent_df['colour'] == 'red').astype(int)
    recent_df['is_green']  = (recent_df['colour'] == 'green').astype(int)
    recent_df['is_violet'] = (recent_df['colour'] == 'violet').astype(int)
    recent_df['num']       = recent_df['number'].astype(int)

    window = recent_df.head(10)
    last_10_big = list(window['is_big'])

    streak = 0
    last_val = last_10_big[-1]
    for v in reversed(last_10_big):
        if v == last_val: streak += 1
        else: break

    big_ratio    = sum(last_10_big) / 10
    last5_ratio  = sum(last_10_big[-5:]) / 5
    alternating  = sum(1 for j in range(1, 10) if last_10_big[j] != last_10_big[j-1]) / 9
    last_nums    = list(window['num'].tail(3))
    avg_num      = window['num'].mean()
    std_num      = window['num'].std()
    p3           = last_10_big[-3:]
    ngram_3      = p3[0]*4 + p3[1]*2 + p3[2]
    violet_count = window['is_violet'].tail(5).sum()
    zero_count   = (window['num'].tail(5) == 0).sum()
    mean_dev     = avg_num - 4.5

    x_pred = np.array([[
        streak, big_ratio, last5_ratio, alternating,
        *last_10_big,
        ngram_3, violet_count, zero_count,
        avg_num, std_num, mean_dev,
        last_nums[0], last_nums[1], last_nums[2]
    ]])

    # Ensemble: RF 60% + GB 40%
    rf_prob = rf.predict_proba(x_pred)[0][1]  # prob of BIG
    gb_prob = gb.predict_proba(x_pred)[0][1]
    ensemble_prob = 0.6 * rf_prob + 0.4 * gb_prob

    prediction = "BIG" if ensemble_prob >= 0.5 else "SMALL"
    confidence = round(max(ensemble_prob, 1 - ensemble_prob) * 100, 1)
    colour = get_colour_prediction(ensemble_prob)

    reasoning = (
        f"RF={round(rf_prob*100)}% BIG | "
        f"GB={round(gb_prob*100)}% BIG | "
        f"Streak={streak} {'BIG' if last_val else 'SMALL'} | "
        f"Big ratio last 10={round(big_ratio*100)}%"
    )

    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        colour_prediction=colour,
        model_used=f"RF+GB Ensemble ({len(X)} samples)",
        features_used=len(x_pred[0]),
        reasoning=reasoning
    )
