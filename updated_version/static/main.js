async function getPrediction() {
    const decisionEl = document.getElementById("decision");
    const detailsEl  = document.getElementById("details");

    decisionEl.innerText = "Loading...";
    detailsEl.innerText  = "";

    // Build the payload from the current time and a default line
    const now    = new Date();
    const days   = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];
    const day    = days[now.getDay()];
    const hour   = now.getHours();
    const seasons = {
        0:"Winter", 1:"Winter", 2:"Spring", 3:"Spring", 4:"Spring",
        5:"Summer", 6:"Summer", 7:"Summer", 8:"Fall", 9:"Fall",
        10:"Fall",  11:"Winter"
    };
    const season = seasons[now.getMonth()];

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                line:    "Red",
                day:     day,
                hour:    hour,
                season:  season,
                station: "Park Street",
            })
        });

        const data = await res.json();

        // Pick a decision label based on risk
        const decisions = {
            "Low":      "LEAVE NOW ✅",
            "Moderate": "LEAVE SOON ⚠️",
            "High":     "EXPECT DELAYS 🔴",
            "Severe":   "AVOID IF POSSIBLE 🚨",
        };
        decisionEl.innerText = decisions[data.risk_level] || "—";

        detailsEl.innerText =
            `Delay Probability: ${data.delay_probability}% • Est. Delay: ${data.delay_minutes} min`;

        // Color logic
        const colors = {
            "Low":      "text-4xl font-bold text-green-400",
            "Moderate": "text-4xl font-bold text-yellow-400",
            "High":     "text-4xl font-bold text-orange-400",
            "Severe":   "text-4xl font-bold text-red-400",
        };
        decisionEl.className = colors[data.risk_level] || "text-4xl font-bold";

        // Show alert if there is one
        if (data.alert_texts && data.alert_texts.length > 0) {
            detailsEl.innerText += ` • ⚠️ ${data.alert_texts[0]}`;
        }

    } catch (err) {
        decisionEl.innerText = "Error";
        detailsEl.innerText  = "Could not fetch data";
    }
}