# prediction_logger.py
# Lightweight learning/calibration logger.
# It does NOT self-modify trading logic. It records forecast outcomes
# and provides rolling reliability by market state.

import json
import os
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "prediction_history.jsonl")
STATS_FILE = os.path.join(LOG_DIR, "prediction_state_stats.json")


def _ensure():
    os.makedirs(LOG_DIR, exist_ok=True)


def log_prediction(
    candle_key,
    forecast,
    state,
    price,
    score,
    confidence,
):
    _ensure()

    row = {
        "type": "prediction",
        "time": datetime.now().isoformat(),
        "candle_key": str(candle_key),
        "forecast": str(forecast),
        "state": str(state),
        "price": float(price),
        "score": float(score),
        "confidence": int(confidence),
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def log_outcome(
    candle_key,
    forecast,
    state,
    predicted_price,
    actual_close,
):
    _ensure()

    move = float(actual_close) - float(predicted_price)

    if forecast == "UP":
        correct = move > 0
    elif forecast == "DOWN":
        correct = move < 0
    else:
        correct = None

    row = {
        "type": "outcome",
        "time": datetime.now().isoformat(),
        "candle_key": str(candle_key),
        "forecast": str(forecast),
        "state": str(state),
        "predicted_price": float(predicted_price),
        "actual_close": float(actual_close),
        "move": move,
        "correct": correct,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    if correct is not None:
        _update_stats(state, correct)


def _update_stats(state, correct):
    _ensure()

    stats = {}

    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                stats = json.load(f)
        except Exception:
            stats = {}

    s = stats.get(
        state,
        {
            "total": 0,
            "correct": 0,
        },
    )

    s["total"] += 1

    if bool(correct):
        s["correct"] += 1

    stats[state] = s

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def get_state_reliability(state):
    """
    Returns reliability in [0.35, 0.75].
    Until >=10 resolved samples exist, neutral 0.50 is returned.
    """

    _ensure()

    if not os.path.exists(STATS_FILE):
        return 0.50

    try:
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except Exception:
        return 0.50

    s = stats.get(state)

    if not s:
        return 0.50

    total = int(s.get("total", 0))
    correct = int(s.get("correct", 0))

    if total < 10:
        return 0.50

    raw = correct / max(total, 1)

    return max(0.35, min(0.75, raw))
