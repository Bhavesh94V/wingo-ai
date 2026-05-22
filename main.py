import os
import json
import asyncio
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from supabase import create_client, Client
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import chisquare
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    print("WARNING: httpx not installed — /live endpoint will use Supabase fallback only")
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Wingo AI Predictor v4.0", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://npbpjsdxisdutcruwkgr.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5wYnBqc2R4aXNkdXRjcnV3a2dyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzk0MzMxOTEsImV4cCI6MjA5NTAwOTE5MX0.Td38AsexT9B7C6LSuFBml3QVFaaMn-m-rcJgXtQ_uIU")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── WinGo colour rules ──────────────────────────────────────────
NUMBER_COLOUR = {
    0: ["red","violet"], 1: ["green"],     2: ["red"],
    3: ["green"],        4: ["red"],        5: ["green","violet"],
    6: ["red"],          7: ["green"],      8: ["red"],  9: ["green"]
}

def get_colour_from_numbers(top_numbers: list) -> str:
    if not top_numbers: return "Red"
    votes = {"red": 0, "green": 0, "violet": 0}
    for n in top_numbers:
        for c in NUMBER_COLOUR.get(n, ["red"]): votes[c] += 1
    res = []
    if votes["green"] > votes["red"]:   res.append("Green")
    elif votes["red"] > votes["green"]: res.append("Red")
    else:                               res.append("Red/Green")
    if votes["violet"] > 0:             res.append("Violet")
    return "+".join(res)

# ── Models ──────────────────────────────────────────────────────
class RoundData(BaseModel):
    number: int
    colour: str
    big_small: str

class PredictRequest(BaseModel):
    game_code: str
    period: Optional[str] = ""
    recent_rounds: List[RoundData]
    panel_prediction: Optional[str] = ""
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
    combined_signal: str
    bias_detected: Optional[bool] = False
    bias_direction: Optional[str] = ""

# ── Supabase helpers ─────────────────────────────────────────────
def fetch_history(game_code: str, limit: int = 600):
    try:
        res = supabase.table("rounds").select("*").eq("game_code", game_code)\
            .order("created_at", desc=True).limit(limit).execute()
        return res.data
    except Exception as e:
        print(f"Fetch error: {e}")
        return []

def store_ai_prediction(period, game_code, prediction, confidence, colour_pred,
                        top_numbers, model_used, panel_pred, panel_conf,
                        combined_conf=None, combined_signal=None,
                        bias_detected=False, bias_direction="", reasoning=""):
    try:
        match_panel = (prediction == panel_pred) if panel_pred else None
        supabase.table("ai_predictions").insert({
            "period": period, "game_code": game_code, "prediction": prediction,
            "confidence": confidence, "colour_pred": colour_pred, "top_numbers": top_numbers,
            "model_used": model_used, "panel_pred": panel_pred or None,
            "panel_conf": panel_conf or None, "match_panel": match_panel,
            "combined_confidence": combined_conf or confidence,
            "combined_signal": combined_signal or "MODERATE",
            "bias_detected": bias_detected,
            "bias_direction": bias_direction or "",
            "reasoning": (reasoning or "")[:400],
        }).execute()
    except Exception as e:
        print(f"Store error: {e}")

def fetch_accuracy_stats(game_code: str) -> dict:
    try:
        res = supabase.table("ai_predictions").select("ai_correct,panel_correct,match_panel")\
            .eq("game_code", game_code).not_.is_("ai_correct","null")\
            .order("created_at", desc=True).limit(100).execute()
        rows = res.data or []
        if not rows: return {"ai_acc":0,"panel_acc":0,"agreement_acc":0,"total":0}
        ai_correct    = sum(1 for r in rows if r.get("ai_correct"))
        panel_correct = sum(1 for r in rows if r.get("panel_correct"))
        agreed        = [r for r in rows if r.get("match_panel")]
        agreed_correct= sum(1 for r in agreed if r.get("ai_correct"))
        return {
            "ai_acc":    round(ai_correct/len(rows)*100,1),
            "panel_acc": round(panel_correct/len(rows)*100,1),
            "agreement_acc": round(agreed_correct/len(agreed)*100,1) if agreed else 0,
            "total": len(rows)
        }
    except: return {"ai_acc":0,"panel_acc":0,"agreement_acc":0,"total":0}

# ── Chi-Square Bias Detection ────────────────────────────────────
def detect_bias(history: list) -> dict:
    """Detect if platform RNG is biased using Chi-square test"""
    if len(history) < 100:
        return {"is_biased": False, "big_pct": 50, "small_pct": 50,
                "red_pct": 50, "green_pct": 50, "p_value": 1.0, "bias_direction": ""}
    try:
        big   = sum(1 for r in history if r.get("big_small","") == "Big")
        small = len(history) - big
        red   = sum(1 for r in history if "red"   in str(r.get("colour","")).lower())
        green = sum(1 for r in history if "green" in str(r.get("colour","")).lower())
        n = len(history)
        _, p_bs  = chisquare([big, small], [n/2, n/2])
        _, p_col = chisquare([red, green], [n/2, n/2])
        biased = p_bs < 0.05 or p_col < 0.05
        direction = ""
        if biased:
            if p_bs < 0.05: direction = "BIG" if big > small else "SMALL"
            if p_col < 0.05: direction += (" " if direction else "") + ("GREEN" if green > red else "RED")
        return {
            "is_biased": biased, "p_value_bs": round(p_bs,4), "p_value_col": round(p_col,4),
            "big_pct": round(big/n*100,1), "small_pct": round(small/n*100,1),
            "red_pct": round(red/n*100,1), "green_pct": round(green/n*100,1),
            "bias_direction": direction.strip(), "samples": n
        }
    except Exception as e:
        return {"is_biased": False, "bias_direction": "", "error": str(e)}

# ── Advanced Feature Engineering (32 features) ──────────────────
def build_features(rows: list):
    if len(rows) < 20:
        return pd.DataFrame(), [], []
    df = pd.DataFrame(rows)
    df = df.sort_values("created_at").reset_index(drop=True)
    df['is_big']    = (df['big_small'] == 'Big').astype(int)
    df['num']       = df['number'].astype(int)
    df['is_red']    = df['colour'].str.contains('red',    case=False, na=False).astype(int)
    df['is_green']  = df['colour'].str.contains('green',  case=False, na=False).astype(int)
    df['is_violet'] = df['colour'].str.contains('violet', case=False, na=False).astype(int)

    features, labels, num_labels = [], [], []
    for i in range(20, len(df) - 1):
        w   = df.iloc[i-20:i]
        tgt = df.iloc[i+1]
        lb  = list(w['is_big'])
        nm  = list(w['num'])

        # Streak
        streak, lv = 0, lb[-1]
        for v in reversed(lb):
            if v == lv: streak += 1
            else: break

        # Multi-window ratios
        r3  = sum(lb[-3:])  / 3
        r5  = sum(lb[-5:])  / 5
        r10 = sum(lb[-10:]) / 10
        r20 = sum(lb)       / 20

        # Alternation rate
        alt = sum(1 for j in range(1,20) if lb[j] != lb[j-1]) / 19

        # N-gram patterns (3 + 4-gram)
        p3 = lb[-3:];   ng3 = p3[0]*4  + p3[1]*2  + p3[2]
        p4 = lb[-4:];   ng4 = p4[0]*8  + p4[1]*4  + p4[2]*2 + p4[3]

        # Transition probabilities
        bb = sum(1 for j in range(len(lb)-1) if lb[j]==1 and lb[j+1]==1)
        bs = sum(1 for j in range(len(lb)-1) if lb[j]==1 and lb[j+1]==0)
        sb = sum(1 for j in range(len(lb)-1) if lb[j]==0 and lb[j+1]==1)
        ss = sum(1 for j in range(len(lb)-1) if lb[j]==0 and lb[j+1]==0)
        bb_rate = bb / max(bb+bs, 1)
        ss_rate = ss / max(sb+ss, 1)

        # Number stats
        avg_n = np.mean(nm);  std_n = np.std(nm)
        skew  = (sum(1 for n in nm if n > 4.5) - 10) / 10
        vol3  = np.std(nm[-3:]);  vol10 = np.std(nm[-10:])
        vol_r = vol3 / max(vol10, 0.001)

        # Colour features
        rc5  = w['is_red'].tail(5).sum()
        gc5  = w['is_green'].tail(5).sum()
        vc5  = w['is_violet'].tail(5).sum()
        col_change = sum(1 for j in range(1,20)
                         if w['is_red'].iloc[j] != w['is_red'].iloc[j-1]) / 19

        # Overdue / zero counts
        z5   = (w['num'].tail(5) == 0).sum()
        lo3  = lb[-3:]
        last_odd = int(nm[-1] % 2 != 0)

        feat = [
            streak, r3, r5, r10, r20, alt,
            ng3, ng4, bb_rate, ss_rate,
            avg_n, std_n, skew, vol_r,
            nm[-3], nm[-2], nm[-1],
            rc5, gc5, vc5, col_change,
            z5, last_odd,
            *lb[-9:]   # last 9 raw big/small (total = 32)
        ]
        features.append(feat)
        labels.append(int(tgt['is_big']))
        num_labels.append(int(tgt['num']))

    return pd.DataFrame(features), labels, num_labels

def build_one_feature(rounds_data: list):
    df = pd.DataFrame(rounds_data)
    df['is_big']    = (df['big_small'] == 'Big').astype(int)
    df['is_red']    = df['colour'].str.contains('red',    case=False, na=False).astype(int)
    df['is_green']  = df['colour'].str.contains('green',  case=False, na=False).astype(int)
    df['is_violet'] = df['colour'].str.contains('violet', case=False, na=False).astype(int)
    df['num']       = df['number'].astype(int)

    lb = list(df['is_big'].head(20))
    nm = list(df['num'].head(20))
    while len(lb) < 20: lb.append(0)
    while len(nm) < 20: nm.append(4)

    streak, lv = 0, lb[-1]
    for v in reversed(lb):
        if v == lv: streak += 1
        else: break

    r3  = sum(lb[-3:])  / 3
    r5  = sum(lb[-5:])  / 5
    r10 = sum(lb[-10:]) / 10
    r20 = sum(lb)       / 20
    alt = sum(1 for j in range(1,20) if lb[j] != lb[j-1]) / 19

    p3 = lb[-3:];   ng3 = p3[0]*4 + p3[1]*2 + p3[2]
    p4 = lb[-4:];   ng4 = p4[0]*8 + p4[1]*4 + p4[2]*2 + p4[3]

    bb = sum(1 for j in range(len(lb)-1) if lb[j]==1 and lb[j+1]==1)
    bs = sum(1 for j in range(len(lb)-1) if lb[j]==1 and lb[j+1]==0)
    sb = sum(1 for j in range(len(lb)-1) if lb[j]==0 and lb[j+1]==1)
    ss = sum(1 for j in range(len(lb)-1) if lb[j]==0 and lb[j+1]==0)
    bb_rate = bb / max(bb+bs, 1)
    ss_rate = ss / max(sb+ss, 1)

    avg_n = np.mean(nm);  std_n = np.std(nm)
    skew  = (sum(1 for n in nm if n > 4.5) - 10) / 10
    vol3  = np.std(nm[-3:]);  vol10 = np.std(nm[-10:])
    vol_r = vol3 / max(vol10, 0.001)

    rc5 = sum(df['is_red'].head(5));  gc5 = sum(df['is_green'].head(5))
    vc5 = sum(df['is_violet'].head(5))
    col_change = sum(1 for j in range(1,min(20,len(df)))
                     if df['is_red'].iloc[j] != df['is_red'].iloc[j-1]) / 19
    z5 = sum(1 for n in nm[-5:] if n == 0)
    last_odd = int(nm[-1] % 2 != 0)

    feat = np.array([[
        streak, r3, r5, r10, r20, alt,
        ng3, ng4, bb_rate, ss_rate,
        avg_n, std_n, skew, vol_r,
        nm[-3], nm[-2], nm[-1],
        rc5, gc5, vc5, col_change,
        z5, last_odd,
        *lb[-9:]
    ]])
    return feat, streak, lv, r10

def predict_top_numbers(big_small, num_model, x_pred):
    try:
        num_probs = num_model.predict_proba(x_pred)[0]
        classes   = num_model.classes_
        filtered  = [(int(classes[i]), float(num_probs[i])) for i in range(len(classes))
                     if (big_small == "BIG" and classes[i] >= 5) or
                        (big_small == "SMALL" and classes[i] <= 4)]
        if not filtered:
            filtered = [(int(classes[i]), float(num_probs[i])) for i in range(len(classes))]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in filtered[:3]]
    except:
        return [5,7,6] if big_small == "BIG" else [1,3,4]

# ── Endpoints ────────────────────────────────────────────────────
@app.get("/")
def root(): return {"status": "Wingo AI Server v4.0 ✅", "xgboost": HAS_XGB}

@app.get("/health")
def health(): return {"status": "ok", "version": "4.0.0", "xgboost": HAS_XGB}

@app.get("/accuracy/{game_code}")
def get_accuracy(game_code: str): return fetch_accuracy_stats(game_code)

@app.get("/bias/{game_code}")
def get_bias(game_code: str):
    history = fetch_history(game_code, limit=500)
    return detect_bias(history)

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    game_code  = req.game_code
    period     = req.period or ""
    panel_pred = (req.panel_prediction or "").upper()
    panel_conf = req.panel_confidence or 0.0

    history = fetch_history(game_code, limit=600)
    if len(history) < 30:
        history = [{"number":r.number,"colour":r.colour,"big_small":r.big_small,
                    "created_at":f"2024-01-01T00:00:{str(i).zfill(2)}Z","game_code":game_code}
                   for i,r in enumerate(reversed(req.recent_rounds))]

    # Bias detection (run on full history)
    bias_info = detect_bias(history)

    result = build_features(history)
    if isinstance(result[0], pd.DataFrame) and result[0].empty:
        return PredictResponse(period=period, prediction="BIG", confidence=50.0,
            colour_prediction="Red", top_numbers=[5,7,6], model_used="fallback",
            features_used=0, reasoning="Need 50+ rounds", panel_agrees=False,
            combined_confidence=50.0, combined_signal="SPLIT")

    X, y, y_num = result
    if len(X) < 20:
        return PredictResponse(period=period, prediction="BIG", confidence=50.0,
            colour_prediction="Red", top_numbers=[5,7,6], model_used="fallback",
            features_used=0, reasoning=f"Only {len(X)} samples", panel_agrees=False,
            combined_confidence=50.0, combined_signal="SPLIT")

    # ── Time-series train/val split ────────────────────────────
    split = int(len(X) * 0.85)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y[:split], y[split:]

    # ── Train RF ──────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                min_samples_leaf=3, random_state=42)
    rf.fit(X_tr, y_tr)

    # ── Train GB ──────────────────────────────────────────────
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    max_depth=4, random_state=42)
    gb.fit(X_tr, y_tr)

    # ── Train XGBoost (free, pip install xgboost) ─────────────
    if HAS_XGB and len(X_tr) > 30:
        xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric='logloss', verbosity=0, random_state=42)
        xgb.fit(X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False)
    else:
        xgb = None

    # ── Calibrated RF (fixes overconfident predictions) ───────
    try:
        cal_rf = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            method='sigmoid', cv=min(3, max(2, len(X_tr)//50))
        )
        cal_rf.fit(X_tr, y_tr)
    except:
        cal_rf = None

    # ── Number predictor ──────────────────────────────────────
    num_rf = RandomForestClassifier(n_estimators=150, max_depth=5,
                                    min_samples_leaf=2, random_state=42)
    try: num_rf.fit(X_tr, y_num[:split])
    except: num_rf = None

    # ── Accuracy stats ─────────────────────────────────────────
    stats = fetch_accuracy_stats(game_code)
    ai_acc    = stats.get("ai_acc", 50)
    panel_acc = stats.get("panel_acc", 50)

    # ── Build prediction feature ───────────────────────────────
    recent_data = [{"number":r.number,"colour":r.colour,"big_small":r.big_small}
                   for r in req.recent_rounds[:21]]
    x_pred, streak, last_val, big_ratio = build_one_feature(recent_data)

    # Probabilities from each model
    rf_prob  = rf.predict_proba(x_pred)[0][1]
    gb_prob  = gb.predict_proba(x_pred)[0][1]
    xgb_prob = xgb.predict_proba(x_pred)[0][1] if xgb else rf_prob
    cal_prob = cal_rf.predict_proba(x_pred)[0][1] if cal_rf else rf_prob

    # ── Bias-aware adjustment ─────────────────────────────────
    bias_boost = 0.0
    if bias_info.get("is_biased"):
        bd = bias_info.get("bias_direction","")
        if "BIG" in bd:   bias_boost = +0.04
        elif "SMALL" in bd: bias_boost = -0.04

    # ── Panel-aware ensemble weighting ────────────────────────
    if panel_pred and panel_acc > ai_acc:
        panel_prob  = 1.0 if panel_pred == "BIG" else 0.0
        if xgb:
            ensemble = 0.25*rf_prob + 0.20*gb_prob + 0.35*xgb_prob + 0.15*cal_prob + 0.05*panel_prob
        else:
            ensemble = 0.30*rf_prob + 0.25*gb_prob + 0.25*cal_prob + 0.20*panel_prob
    else:
        if xgb:
            ensemble = 0.25*rf_prob + 0.20*gb_prob + 0.40*xgb_prob + 0.15*cal_prob
        else:
            ensemble = 0.35*rf_prob + 0.30*gb_prob + 0.35*cal_prob

    ensemble   = min(0.97, max(0.03, ensemble + bias_boost))
    prediction = "BIG" if ensemble >= 0.5 else "SMALL"
    confidence = round(max(ensemble, 1-ensemble) * 100, 1)

    # ── Top numbers ───────────────────────────────────────────
    top_nums = predict_top_numbers(prediction, num_rf, x_pred) if num_rf else (
        [5,7,6] if prediction == "BIG" else [1,3,4])

    # ── Colour from numbers ───────────────────────────────────
    colour_pred = get_colour_from_numbers(top_nums)

    # ── Combined signal ───────────────────────────────────────
    panel_agrees = (panel_pred == prediction) if panel_pred else False
    if panel_agrees and panel_pred:
        boost = min(8.0, panel_conf*0.05 + confidence*0.05)
        combined_conf = round(min(95.0, confidence + boost), 1)
        combined_signal = "STRONG" if combined_conf >= 70 else "MODERATE"
    elif panel_pred and not panel_agrees:
        combined_conf   = round(max(50.0, confidence - 5.0), 1)
        combined_signal = "SPLIT"
    else:
        combined_conf   = confidence
        combined_signal = "MODERATE" if confidence >= 65 else "WEAK"

    model_tag = f"RF+GB+{'XGB+' if xgb else ''}Cal ({len(X)} samples)"
    reasoning = (
        f"RF={round(rf_prob*100)}% | GB={round(gb_prob*100)}% | "
        f"XGB={round(xgb_prob*100)}% | Cal={round(cal_prob*100)}% | "
        f"Streak={streak} {'BIG' if last_val else 'SMALL'} | "
        f"Big%={round(big_ratio*100)}% | "
        f"{'⚠️Bias:'+bias_info.get('bias_direction','') if bias_info.get('is_biased') else 'RNG:Fair'} | "
        f"AIacc={ai_acc}% PanelAcc={panel_acc}%"
    )

    # ── Store async ───────────────────────────────────────────
    asyncio.create_task(asyncio.to_thread(
        store_ai_prediction, period, game_code, prediction,
        confidence, colour_pred, top_nums, model_tag, panel_pred, panel_conf,
        combined_conf, combined_signal,
        bias_info.get("is_biased", False), bias_info.get("bias_direction", ""),
        reasoning
    ))

    return PredictResponse(
        period=period, prediction=prediction, confidence=confidence,
        colour_prediction=colour_pred, top_numbers=top_nums,
        model_used=model_tag, features_used=len(x_pred[0]),
        reasoning=reasoning, panel_agrees=panel_agrees,
        combined_confidence=combined_conf, combined_signal=combined_signal,
        bias_detected=bias_info.get("is_biased", False),
        bias_direction=bias_info.get("bias_direction", "")
    )

@app.post("/update_actual")
async def update_actual(data: dict):
    try:
        period     = data.get("period")
        actual_num = data.get("actual_number")
        actual_col = data.get("actual_colour")
        actual_bs  = "BIG" if actual_num >= 5 else "SMALL"
        res = supabase.table("ai_predictions").select("id,prediction,panel_pred")\
            .eq("period", period).execute()
        rows = res.data or []
        for row in rows:
            supabase.table("ai_predictions").update({
                "actual_result": actual_bs,
                "actual_number": actual_num,
                "actual_colour": actual_col,
                "ai_correct":    (row["prediction"] == actual_bs) if row.get("prediction") else None,
                "panel_correct": (row["panel_pred"] == actual_bs) if row.get("panel_pred") else None
            }).eq("id", row["id"]).execute()
        return {"updated": len(rows)}
    except Exception as e:
        return {"error": str(e)}

# ── /live endpoint: server-side fetch from WinGo draw API ────────
@app.get("/live/{game_code}")
async def live_predict(game_code: str):
    """
    Standalone endpoint for PWA / mobile use.
    Fetches latest rounds from WinGo draw API server-side (no CORS),
    stores new rounds in Supabase, returns full prediction.
    """
    rounds_raw = []

    # 1. Try WinGo public draw API (no auth needed)
    if HAS_HTTPX:
        draw_urls = [
            f"https://draw.ar-lottery01.com/WinGo/{game_code}.json",
            f"https://draw.ar-lottery01.com/WinGo/{game_code}/GetHistoryIssuePage.json",
        ]
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                for url in draw_urls:
                    try:
                        r = await client.get(url, params={"ts": int(time.time()*1000)})
                        if r.status_code == 200:
                            data = r.json()
                            items = (data.get("data") or data.get("list") or
                                     (data if isinstance(data, list) else []))
                            if items:
                                rounds_raw = items[:30]
                                break
                    except:
                        continue
        except:
            pass

    # 2. Fall back to Supabase if draw API unreachable
    if not rounds_raw:
        db = fetch_history(game_code, limit=50)
        if db:
            rounds_raw = [{"issueNumber": r.get("period",""), "number": r.get("number",0),
                            "colour": r.get("colour","red"), "big_small": r.get("big_small","Small")}
                          for r in db[:30]]

    if not rounds_raw:
        return {"error": "No data available", "game_code": game_code}

    # 3. Parse rounds
    def parse_colour(num: int, colour_str: str) -> str:
        c = (colour_str or "").lower()
        if c: return c
        mapping = {0:"red,violet",1:"green",2:"red",3:"green",4:"red",
                   5:"green,violet",6:"red",7:"green",8:"red",9:"green"}
        return mapping.get(num, "red")

    rounds = []
    for item in rounds_raw:
        try:
            num = int(item.get("number", 0))
            col = parse_colour(num, item.get("colour") or item.get("color",""))
            bs  = "Big" if num >= 5 else "Small"
            rounds.append({"number": num, "colour": col, "big_small": bs,
                           "issueNumber": str(item.get("issueNumber",""))})
        except:
            continue

    if len(rounds) < 10:
        return {"error": f"Not enough rounds: {len(rounds)}", "game_code": game_code}

    # 4. Store new rounds in Supabase
    try:
        existing_periods = {r.get("period","") for r in fetch_history(game_code, limit=50)}
        new_rows = [{"period": r["issueNumber"], "number": r["number"],
                     "colour": r["colour"], "big_small": r["big_small"],
                     "game_code": game_code}
                    for r in rounds if r["issueNumber"] and r["issueNumber"] not in existing_periods]
        if new_rows:
            supabase.table("rounds").insert(new_rows).execute()
    except: pass

    # 5. Current period = next after latest
    try:
        latest_period = rounds[0]["issueNumber"]
        next_period = str(int(latest_period) + 1) if latest_period.isdigit() else ""
    except:
        next_period = ""

    # 6. Run prediction using the rounds as recent_rounds
    recent = [RoundData(number=r["number"], colour=r["colour"], big_small=r["big_small"])
              for r in rounds[:30]]
    req = PredictRequest(game_code=game_code, period=next_period, recent_rounds=recent)

    pred_result = await predict(req)

    # 7. Bias detection
    full_history = fetch_history(game_code, limit=300)
    bias = detect_bias(full_history) if full_history else {"is_biased": False}

    # 8. Accuracy stats
    stats = fetch_accuracy_stats(game_code)

    return {
        "game_code":        game_code,
        "current_period":   next_period,
        "latest_number":    rounds[0]["number"],
        "latest_colour":    rounds[0]["colour"],
        "recent_rounds":    rounds[:10],
        "prediction":       pred_result.dict(),
        "bias":             bias,
        "accuracy":         stats,
        "data_source":      "draw_api" if rounds_raw and not all(r.get("period","") in [""] for r in rounds_raw[:1]) else "supabase"
    }

