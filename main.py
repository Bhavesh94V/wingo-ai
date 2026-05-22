import os
import json
import asyncio
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

app = FastAPI(title="Wingo AI Predictor", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://npbpjsdxisdutcruwkgr.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5wYnBqc2R4aXNkdXRjcnV3a2dyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzk0MzMxOTEsImV4cCI6MjA5NTAwOTE5MX0.Td38AsexT9B7C6LSuFBml3QVFaaMn-m-rcJgXtQ_uIU")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── WinGo colour rules ──────────────────────────────────────────
# 0→red+violet | 1→green | 2→red | 3→green | 4→red
# 5→green+violet | 6→red | 7→green | 8→red | 9→green
NUMBER_COLOUR = {
    0: ["red", "violet"], 1: ["green"],      2: ["red"],
    3: ["green"],         4: ["red"],         5: ["green", "violet"],
    6: ["red"],           7: ["green"],       8: ["red"],   9: ["green"]
}

def get_colour_from_numbers(top_numbers: list) -> str:
    """Determine most likely colour from predicted top numbers"""
    if not top_numbers:
        return "Red"
    colour_votes = {"red": 0, "green": 0, "violet": 0}
    for n in top_numbers:
        for c in NUMBER_COLOUR.get(n, ["red"]):
            colour_votes[c] += 1
    # Build result string
    result = []
    if colour_votes["green"] > colour_votes["red"]:
        result.append("Green")
    elif colour_votes["red"] > colour_votes["green"]:
        result.append("Red")
    else:
        result.append("Red/Green")
    if colour_votes["violet"] > 0:
        result.append("Violet")
    return "+".join(result) if result else "Red"

# ── Models ──────────────────────────────────────────────────────
class RoundData(BaseModel):
    number: int
    colour: str
    big_small: str

class PredictRequest(BaseModel):
    game_code: str
    period: Optional[str] = ""
    recent_rounds: List[RoundData]
    # Panel prediction for cross-learning
    panel_prediction: Optional[str] = ""    # "BIG" or "SMALL"
    panel_confidence: Optional[float] = 0.0
    panel_method: Optional[str] = ""

class PredictResponse(BaseModel):
    period: str
    prediction: str
    confidence: float
    colour_prediction: str
    top_numbers: List[int]
    model_used: str
    features_used: int
    reasoning: str
    panel_agrees: bool
    combined_confidence: float
    combined_signal: str   # "STRONG" / "MODERATE" / "SPLIT"

# ── Supabase helpers ─────────────────────────────────────────────
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

def store_ai_prediction(period: str, game_code: str, prediction: str,
                        confidence: float, colour_pred: str, top_numbers: list,
                        model_used: str, panel_pred: str, panel_conf: float):
    """Store AI prediction in ai_predictions table"""
    try:
        match_panel = (prediction == panel_pred) if panel_pred else None
        supabase.table("ai_predictions").insert({
            "period":       period,
            "game_code":    game_code,
            "prediction":   prediction,
            "confidence":   confidence,
            "colour_pred":  colour_pred,
            "top_numbers":  top_numbers,
            "model_used":   model_used,
            "panel_pred":   panel_pred or None,
            "panel_conf":   panel_conf or None,
            "match_panel":  match_panel
        }).execute()
    except Exception as e:
        print(f"Store AI pred error: {e}")

def fetch_accuracy_stats(game_code: str) -> dict:
    """Fetch accuracy of AI vs panel from last 100 completed rounds"""
    try:
        res = supabase.table("ai_predictions")\
            .select("ai_correct,panel_correct,match_panel")\
            .eq("game_code", game_code)\
            .not_.is_("ai_correct", "null")\
            .order("created_at", desc=True)\
            .limit(100)\
            .execute()
        rows = res.data or []
        if not rows:
            return {"ai_acc": 0, "panel_acc": 0, "agreement_acc": 0, "total": 0}
        ai_correct    = sum(1 for r in rows if r.get("ai_correct"))
        panel_correct = sum(1 for r in rows if r.get("panel_correct"))
        agreed        = [r for r in rows if r.get("match_panel")]
        agreed_correct= sum(1 for r in agreed if r.get("ai_correct"))
        return {
            "ai_acc":       round(ai_correct / len(rows) * 100, 1),
            "panel_acc":    round(panel_correct / len(rows) * 100, 1),
            "agreement_acc":round(agreed_correct / len(agreed) * 100, 1) if agreed else 0,
            "total":        len(rows)
        }
    except:
        return {"ai_acc": 0, "panel_acc": 0, "agreement_acc": 0, "total": 0}

# ── Feature engineering ──────────────────────────────────────────
def build_features(rows: list):
    if len(rows) < 15:
        return pd.DataFrame(), [], []

    df = pd.DataFrame(rows)
    df = df.sort_values("created_at").reset_index(drop=True)
    df['is_big']    = (df['big_small'] == 'Big').astype(int)
    df['num']       = df['number'].astype(int)
    df['is_red']    = df['colour'].str.contains('red',    case=False, na=False).astype(int)
    df['is_green']  = df['colour'].str.contains('green',  case=False, na=False).astype(int)
    df['is_violet'] = df['colour'].str.contains('violet', case=False, na=False).astype(int)

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
        red_cnt     = window['is_red'].tail(5).sum()
        green_cnt   = window['is_green'].tail(5).sum()
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
    df = pd.DataFrame(rounds_data)
    df['is_big']    = (df['big_small'] == 'Big').astype(int)
    df['is_red']    = df['colour'].str.contains('red',    case=False, na=False).astype(int)
    df['is_green']  = df['colour'].str.contains('green',  case=False, na=False).astype(int)
    df['is_violet'] = df['colour'].str.contains('violet', case=False, na=False).astype(int)
    df['num']       = df['number'].astype(int)

    window       = df.head(10)
    last_10_big  = list(window['is_big'])
    streak       = 0
    last_val     = last_10_big[-1]
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

def predict_top_numbers(big_small: str, num_model, x_pred) -> list:
    try:
        num_probs = num_model.predict_proba(x_pred)[0]
        classes   = num_model.classes_
        filtered  = [(int(classes[i]), float(num_probs[i])) for i in range(len(classes))
                     if (big_small == "BIG"   and classes[i] >= 5) or
                        (big_small == "SMALL" and classes[i] <= 4)]
        if not filtered:
            filtered = [(int(classes[i]), float(num_probs[i])) for i in range(len(classes))]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in filtered[:3]]
    except:
        return [5, 7, 6] if big_small == "BIG" else [1, 3, 4]

# ── Endpoints ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Wingo AI Server Running ✅", "version": "3.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0"}

@app.get("/accuracy/{game_code}")
def get_accuracy(game_code: str):
    return fetch_accuracy_stats(game_code)

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    game_code    = req.game_code
    period       = req.period or ""
    panel_pred   = (req.panel_prediction or "").upper()
    panel_conf   = req.panel_confidence or 0.0

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

    # Fallback if not enough data
    if isinstance(result[0], pd.DataFrame) and result[0].empty:
        return PredictResponse(
            period=period, prediction="BIG", confidence=50.0,
            colour_prediction="Red", top_numbers=[5,7,6],
            model_used="fallback", features_used=0,
            reasoning="Need 50+ rounds of data.",
            panel_agrees=False, combined_confidence=50.0, combined_signal="SPLIT"
        )

    X, y, y_num = result

    if len(X) < 20:
        return PredictResponse(
            period=period, prediction="BIG", confidence=50.0,
            colour_prediction="Red", top_numbers=[5,7,6],
            model_used="fallback", features_used=0,
            reasoning=f"Only {len(X)} samples. Need more data.",
            panel_agrees=False, combined_confidence=50.0, combined_signal="SPLIT"
        )

    # ── Train models ──────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                min_samples_leaf=3, random_state=42)
    rf.fit(X, y)

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    max_depth=4, random_state=42)
    gb.fit(X, y)

    num_rf = RandomForestClassifier(n_estimators=150, max_depth=5,
                                    min_samples_leaf=2, random_state=42)
    try:
        num_rf.fit(X, y_num)
    except:
        num_rf = None

    # ── Accuracy stats (async doesn't block) ─────────────
    stats = fetch_accuracy_stats(game_code)
    ai_acc    = stats.get("ai_acc", 50)
    panel_acc = stats.get("panel_acc", 50)

    # ── Build prediction feature ──────────────────────────
    recent_data = [
        {"number": r.number, "colour": r.colour, "big_small": r.big_small}
        for r in req.recent_rounds[:11]
    ]
    x_pred, streak, last_val, big_ratio = build_one_feature(recent_data)

    rf_prob  = rf.predict_proba(x_pred)[0][1]
    gb_prob  = gb.predict_proba(x_pred)[0][1]

    # ── Panel-aware ensemble weighting ───────────────────
    # If panel accuracy > AI accuracy, give panel more weight
    if panel_pred and panel_acc > ai_acc:
        panel_weight = 0.35
        panel_prob   = 1.0 if panel_pred == "BIG" else 0.0
        ensemble     = 0.35 * rf_prob + 0.30 * gb_prob + panel_weight * panel_prob
    else:
        ensemble     = 0.60 * rf_prob + 0.40 * gb_prob

    prediction = "BIG" if ensemble >= 0.5 else "SMALL"
    confidence = round(max(ensemble, 1 - ensemble) * 100, 1)

    # ── Top numbers ───────────────────────────────────────
    top_nums = predict_top_numbers(prediction, num_rf, x_pred) if num_rf else (
        [5,7,6] if prediction == "BIG" else [1,3,4]
    )

    # ── Accurate colour from actual predicted numbers ─────
    colour_pred = get_colour_from_numbers(top_nums)

    # ── Combined signal ───────────────────────────────────
    panel_agrees = (panel_pred == prediction) if panel_pred else False
    if panel_agrees and panel_pred:
        boost = min(8.0, (panel_conf * 0.05 + confidence * 0.05))
        combined_conf = round(min(95.0, confidence + boost), 1)
        combined_signal = "STRONG" if combined_conf >= 70 else "MODERATE"
    elif panel_pred and not panel_agrees:
        # Penalise slightly when they disagree
        combined_conf   = round(max(50.0, confidence - 5.0), 1)
        combined_signal = "SPLIT"
    else:
        combined_conf   = confidence
        combined_signal = "MODERATE" if confidence >= 65 else "WEAK"

    reasoning = (
        f"RF={round(rf_prob*100)}% BIG | "
        f"GB={round(gb_prob*100)}% BIG | "
        f"Streak={streak} {'BIG' if last_val else 'SMALL'} | "
        f"Big%={round(big_ratio*100)}% | "
        f"Samples={len(X)} | "
        f"AIacc={ai_acc}% PanelAcc={panel_acc}%"
    )

    # ── Store prediction async ────────────────────────────
    asyncio.create_task(asyncio.to_thread(
        store_ai_prediction, period, game_code, prediction,
        confidence, colour_pred, top_nums, f"RF+GB+Num ({len(X)} samples)",
        panel_pred, panel_conf
    ))

    return PredictResponse(
        period=period,
        prediction=prediction,
        confidence=confidence,
        colour_prediction=colour_pred,
        top_numbers=top_nums,
        model_used=f"RF+GB+Num ({len(X)} samples)",
        features_used=len(x_pred[0]),
        reasoning=reasoning,
        panel_agrees=panel_agrees,
        combined_confidence=combined_conf,
        combined_signal=combined_signal
    )

@app.post("/update_actual")
async def update_actual(data: dict):
    """Called by browser when actual round result is known"""
    try:
        period       = data.get("period")
        actual_num   = data.get("actual_number")
        actual_col   = data.get("actual_colour")
        actual_bs    = "BIG" if actual_num >= 5 else "SMALL"

        res = supabase.table("ai_predictions")\
            .select("id,prediction,panel_pred")\
            .eq("period", period)\
            .execute()
        rows = res.data or []
        for row in rows:
            ai_correct    = (row["prediction"] == actual_bs) if row.get("prediction") else None
            panel_correct = (row["panel_pred"] == actual_bs) if row.get("panel_pred") else None
            supabase.table("ai_predictions").update({
                "actual_result": actual_bs,
                "actual_number": actual_num,
                "actual_colour": actual_col,
                "ai_correct":    ai_correct,
                "panel_correct": panel_correct
            }).eq("id", row["id"]).execute()
        return {"updated": len(rows)}
    except Exception as e:
        return {"error": str(e)}
