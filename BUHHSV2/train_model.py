import os
import json
import pickle
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

load_dotenv()
API_KEY  = os.getenv("MBTA_API_KEY")
BASE_URL = "https://api-v3.mbta.com"
HEADERS  = {"x-api-key": API_KEY}

np.random.seed(42)

ROUTE_IDS = {
    "Red":    "Red",
    "Orange": "Orange",
    "Green":  "Green-B,Green-C,Green-D,Green-E",
    "Blue":   "Blue",
    "Silver": "741,742,743,746",
}

HIGH_DELAY_STATIONS = {
    "Red":    ["place-harsq", "place-pktrm", "place-dwnxg", "place-jfk"],
    "Orange": ["place-sull",  "place-dwnxg", "place-bbsta", "place-rugg"],
    "Green":  ["place-pktrm", "place-coecl", "place-kencl", "place-gover"],
    "Blue":   ["place-aport", "place-aqucl", "place-gover"],
    "Silver": ["place-sstat", "place-nubn"],
}

LINE_BASE_DELAY = {"Red": 0.38, "Orange": 0.42, "Green": 0.52, "Blue": 0.28, "Silver": 0.35}
SEASON_MULT     = {"Winter": 1.40, "Spring": 1.05, "Summer": 0.95, "Fall": 1.10}


# ── API helpers ───────────────────────────────────────────────────────────────

def fetch_predictions(route_id):
    """Fetch real-time predictions for a route from the MBTA API."""
    url    = f"{BASE_URL}/predictions"
    params = {"filter[route]": route_id, "include": "stop", "page[limit]": 100}
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        print(f"  Warning: could not fetch predictions for {route_id}: {e}")
        return []

def fetch_alerts(route_id):
    """Fetch active alerts/delays for a route."""
    url    = f"{BASE_URL}/alerts"
    params = {"filter[route]": route_id, "filter[activity]": "BOARD"}
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        print(f"  Warning: could not fetch alerts for {route_id}: {e}")
        return []

def fetch_stops(route_id):
    """Fetch all stops for a route."""
    url    = f"{BASE_URL}/stops"
    params = {"filter[route]": route_id}
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        return {s["id"]: s["attributes"]["name"] for s in r.json().get("data", [])}
    except Exception as e:
        print(f"  Warning: could not fetch stops for {route_id}: {e}")
        return {}


# ── Feature helpers ───────────────────────────────────────────────────────────

def get_season(month):
    if month in (12, 1, 2): return "Winter"
    if month in (3, 4, 5):  return "Spring"
    if month in (6, 7, 8):  return "Summer"
    return "Fall"

def is_rush_hour(hour):
    return 1 if (7 <= hour <= 9) or (16 <= hour <= 18) else 0

def is_weekend(day):
    return 1 if day >= 5 else 0  # 5=Saturday, 6=Sunday


# ── Build dataset from API + synthetic augmentation ───────────────────────────

def build_dataset():
    """
    Pull real predictions from the MBTA API and extract delay features.
    Since the API only gives current/near-future predictions, we augment
    with synthetic records to give the model enough data to train on.
    """
    records = []
    now     = pd.Timestamp.now()
    month   = now.month
    season  = get_season(month)
    weekday = now.weekday()
    hour    = now.hour

    print("Fetching real MBTA data...")
    for line, route_id in ROUTE_IDS.items():
        print(f"  Pulling {line} line...")

        stops      = fetch_stops(route_id)
        alerts     = fetch_alerts(route_id)
        has_alert  = len(alerts) > 0
        predictions = fetch_predictions(route_id)

        for pred in predictions:
            attr       = pred.get("attributes", {})
            stop_id    = pred.get("relationships", {}).get("stop", {}).get("data", {}).get("id", "")
            arrival    = attr.get("arrival_time")
            departure  = attr.get("departure_time")
            status     = attr.get("status", "")

            if not arrival and not departure:
                continue

            # Parse hour from prediction timestamp
            time_str = arrival or departure
            try:
                pred_hour = pd.Timestamp(time_str).hour
            except:
                pred_hour = hour

            hotspot   = 1 if stop_id in HIGH_DELAY_STATIONS.get(line, []) else 0
            rush      = is_rush_hour(pred_hour)
            weekend   = is_weekend(weekday)

            # Determine if delayed based on status string or alert
            delayed = 1 if (has_alert or "delay" in status.lower() or "stopped" in status.lower()) else 0

            # Rough delay minutes from status if available
            delay_minutes = 0.0
            if delayed:
                delay_minutes = np.random.uniform(3, 15) if has_alert else np.random.uniform(1, 8)

            records.append({
                "line":          line,
                "hour":          pred_hour,
                "season":        season,
                "weekday":       weekday,
                "is_rush_hour":  rush,
                "is_weekend":    weekend,
                "is_hotspot":    hotspot,
                "has_alert":     int(has_alert),
                "delayed":       delayed,
                "delay_minutes": round(delay_minutes, 1),
            })

    real_count = len(records)
    print(f"  Got {real_count} real prediction records")

    # ── Synthetic augmentation ────────────────────────────────────────────────
    # The live API only gives us a snapshot in time. We augment with synthetic
    # records that simulate the full range of times, days, and seasons so the
    # model learns generalizable patterns rather than just right now.
    print("Augmenting with synthetic records...")
    for _ in range(10000):
        line    = np.random.choice(list(ROUTE_IDS.keys()))
        weekday = np.random.randint(0, 7)
        hour    = np.random.randint(5, 24)
        season  = np.random.choice(["Winter", "Spring", "Summer", "Fall"])
        hotspot = np.random.randint(0, 2)
        rush    = is_rush_hour(hour)
        weekend = is_weekend(weekday)
        alert   = np.random.choice([0, 0, 0, 1])  # 25% chance of alert

        p = LINE_BASE_DELAY[line] * SEASON_MULT[season]
        p += rush    * 0.20
        p -= weekend * 0.08
        p += hotspot * 0.12
        p += alert   * 0.15
        p  = float(np.clip(p, 0.02, 0.95))

        delayed = int(np.random.random() < p)
        if delayed:
            delay_minutes = float(np.clip(np.random.exponential(6) + rush * np.random.uniform(2, 8), 1, 45))
        else:
            delay_minutes = 0.0

        records.append({
            "line":          line,
            "hour":          hour,
            "season":        season,
            "weekday":       weekday,
            "is_rush_hour":  rush,
            "is_weekend":    weekend,
            "is_hotspot":    hotspot,
            "has_alert":     alert,
            "delayed":       delayed,
            "delay_minutes": round(delay_minutes, 1),
        })

    print(f"  Total records: {len(records)} ({real_count} real + {len(records)-real_count} synthetic)")
    return pd.DataFrame(records)


# ── Train ─────────────────────────────────────────────────────────────────────

def train(df):
    le_line    = LabelEncoder().fit(list(ROUTE_IDS.keys()))
    le_season  = LabelEncoder().fit(["Winter", "Spring", "Summer", "Fall"])

    df["line_enc"]   = le_line.transform(df["line"])
    df["season_enc"] = le_season.transform(df["season"])

    FEATURES = ["line_enc", "weekday", "hour", "season_enc",
                "is_rush_hour", "is_weekend", "is_hotspot", "has_alert"]

    X     = df[FEATURES]
    y_cls = df["delayed"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_cls, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    print("\nClassifier report:")
    print(classification_report(y_te, clf.predict(X_te)))

    delayed_mask = df["delayed"] == 1
    reg = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    reg.fit(X[delayed_mask], df.loc[delayed_mask, "delay_minutes"])

    encoders = {"line": le_line, "season": le_season, "features": FEATURES}

    hourly_stats = {}
    for line in ROUTE_IDS.keys():
        sub = df[df["line"] == line]
        hourly_stats[line] = (
            sub.groupby("hour")["delayed"].mean()
               .reindex(range(5, 24), fill_value=0)
               .round(3)
               .to_dict()
        )

    return clf, reg, encoders, hourly_stats


if __name__ == "__main__":
    df = build_dataset()

    print("\nTraining models...")
    clf, reg, encoders, hourly_stats = train(df)

    os.makedirs("static", exist_ok=True)

    payload = {"clf": clf, "reg": reg, "encoders": encoders,
               "hourly_stats": hourly_stats}
    with open("model.pkl", "wb") as f:
        pickle.dump(payload, f)

    with open("static/hourly_stats.json", "w") as f:
        json.dump(hourly_stats, f)

    print("\n✅  model.pkl saved")
    print("✅  static/hourly_stats.json saved")