import os
import pickle
import datetime
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

MBTA_API_KEY = os.getenv("MBTA_API_KEY")
BASE_URL = "https://api-v3.mbta.com"
HEADERS  = {"x-api-key": MBTA_API_KEY}

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(MODEL_PATH, "rb") as f:
    payload = pickle.load(f)

clf          = payload["clf"]
reg          = payload["reg"]
encoders     = payload["encoders"]
hourly_stats = payload["hourly_stats"]

FEATURES = encoders["features"]

ROUTE_IDS = {
    "Red": "Red",
    "Orange": "Orange",
    "Green": "Green-B,Green-C,Green-D,Green-E",
    "Blue": "Blue",
    "Silver": "741,742,743,746",
}

LINE_COLOR = {
    "Red": "#DA291C",
    "Orange": "#ED8B00",
    "Green": "#00843D",
    "Blue": "#003DA5",
    "Silver": "#7C878E",
}

STATIONS = {
    "Red": ["Alewife","Davis","Porter","Harvard","Central","Kendall/MIT",
            "Charles/MGH","Park Street","Downtown Crossing","South Station"],
    "Orange": ["Oak Grove","Malden Center","Wellington","North Station",
               "State","Back Bay","Forest Hills"],
    "Green": ["Lechmere","North Station","Government Center","Park Street",
              "Copley","Kenmore"],
    "Blue": ["Wonderland","Airport","Aquarium","Government Center"],
    "Silver": ["Logan Airport","South Station","Nubian"]
}

HIGH_DELAY_STATIONS = {
    "Red": ["Harvard","Park Street","Downtown Crossing"],
    "Orange": ["Back Bay","Ruggles"],
    "Green": ["Copley","Kenmore"],
    "Blue": ["Airport"],
    "Silver": ["South Station"]
}

def current_season():
    m = datetime.date.today().month
    if m in (12,1,2): return "Winter"
    if m in (3,4,5):  return "Spring"
    if m in (6,7,8):  return "Summer"
    return "Fall"


def get_suggestion(delay, station, line):
    if delay < 3:
        return f"{line} Line is on time — head to the platform."
    elif delay < 8:
        return f"Short delay at {station} — stay nearby."
    elif delay < 15:
        return f"Moderate delay — grab something quick near {station}."
    elif delay < 25:
        return f"Long delay — consider alternate routes."
    else:
        return f"Major delay — consider other transport."


def get_things_to_do(delay, station):
    if delay < 5:
        things = ["Head straight to the platform"]
    elif delay < 10:
        things = ["Grab a quick coffee", "Check messages", "Stretch or walk around"]
    elif delay < 20:
        things = ["Sit down and relax", "Get a snack", "Explore nearby shops"]
    else:
        things = ["Consider Uber or bus alternatives", "Walk part of your route", "Find a place to study"]

    if station == "Harvard":
        things.append("Walk through Harvard Yard")
    if station == "South Station":
        things.append("Grab food in the station hall")
    if station == "Kendall/MIT":
        things.append("Study at a nearby café")

    return things


def fetch_live_predictions(line):
    route_id = ROUTE_IDS.get(line, line)
    try:
        r = requests.get(
            f"{BASE_URL}/predictions",
            headers=HEADERS,
            params={"filter[route]": route_id, "sort": "arrival_time", "page[limit]": 5},
            timeout=5
        )
        data = r.json().get("data", [])
        times = []
        for p in data:
            t = p["attributes"].get("arrival_time") or p["attributes"].get("departure_time")
            if t:
                times.append(t)
        return times[:3]
    except:
        return []


def fetch_alerts(line):
    route_id = ROUTE_IDS.get(line, line)
    try:
        r = requests.get(f"{BASE_URL}/alerts", headers=HEADERS, params={"filter[route]": route_id}, timeout=5)
        return len(r.json().get("data", [])) > 0
    except:
        return False


def encode_input(line, weekday, hour, season, station, has_alert):
    import pandas as pd

    rush    = 1 if (7 <= hour <= 9 or 16 <= hour <= 18) else 0
    weekend = 1 if weekday >= 5 else 0
    hotspot = 1 if station in HIGH_DELAY_STATIONS.get(line, []) else 0

    le = encoders

    vec = {
        "line_enc": le["line"].transform([line])[0],
        "weekday": weekday,
        "hour": hour,
        "season_enc": le["season"].transform([season])[0],
        "is_rush_hour": rush,
        "is_weekend": weekend,
        "is_hotspot": hotspot,
        "has_alert": has_alert,
    }

    return pd.DataFrame([vec])[FEATURES]


@app.route("/")
def index():
    return render_template("index.html", lines=list(ROUTE_IDS.keys()), stations=STATIONS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    line    = data.get("line", "Red")
    station = data.get("station", STATIONS[line][0])

    now = datetime.datetime.now()
    weekday = now.weekday()
    hour = now.hour
    season = current_season()

    has_alert = int(fetch_alerts(line))

    X = encode_input(line, weekday, hour, season, station, has_alert)

    delay_prob = float(clf.predict_proba(X)[0][1])
    delay_min  = float(reg.predict(X)[0]) if delay_prob > 0.4 else 0
    delay_min  = max(0, min(delay_min, 30))

    decision = "LEAVE NOW" if delay_min < 5 else "WAIT"

    risk = "Low" if delay_prob < 0.3 else "Moderate" if delay_prob < 0.6 else "High"

    return jsonify({
        "decision": decision,
        "delay_minutes": round(delay_min,1),
        "delay_probability": round(delay_prob*100,1),
        "risk_level": risk,
        "live_predictions": fetch_live_predictions(line),
        "suggestion": get_suggestion(delay_min, station, line),
        "things_to_do": get_things_to_do(delay_min, station)
    })


@app.route("/vehicles/<line>")
def vehicles(line):
    route_id = ROUTE_IDS.get(line, line)
    try:
        r = requests.get(f"{BASE_URL}/vehicles", headers=HEADERS, params={"filter[route]": route_id}, timeout=5)
        data = r.json().get("data", [])
        return jsonify([
            {"lat": v["attributes"]["latitude"], "lon": v["attributes"]["longitude"]}
            for v in data if v["attributes"].get("latitude")
        ])
    except:
        return jsonify([])


if __name__ == "__main__":
    app.run(debug=True)