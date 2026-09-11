import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Example dataset: [sleep_hours, water_glasses, bench_press_kg]
X = np.array([
    [7.5, 7, 80], [8.0, 8, 82], [6.5, 6, 78], [7.0, 9, 85],
    [9.0, 8, 80], [7.5, 7, 83], [8.0, 8, 84], [6.0, 6, 81],
    [8.5, 9, 85], [7.0, 8, 80], [7.5, 8, 86], [9.0, 7, 79],
    [7.0, 9, 84], [7.5, 8, 83], [7.0, 7, 82], [8.0, 8, 86],
    [6.5, 6, 79], [7.5, 9, 88], [8.0, 8, 81], [7.0, 7, 85],
    [8.5, 9, 87], [7.0, 8, 82], [7.5, 8, 86], [6.5, 6, 80],
    [8.0, 9, 87], [9.5, 7, 79], [7.0, 8, 84], [8.0, 9, 86]
])

# Step counts (target variable)
steps = np.array([
    9200, 10500, 8800, 11000, 7600, 9400, 10200, 8900,
    10800, 9100, 11200, 7900, 10000, 9700, 9500, 10300,
    8600, 11500, 8200, 9800, 10600, 9000, 10100, 8400,
    10900, 7500, 9600, 10400
])

# Convert to binary classification: 1 if >= 10k steps, else 0
y = (steps >= 10000).astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(X_train, y_train)

# Prediction function
def predict_day(sleep_hr, water_glasses, bench_kg):
    """Predict whether today will hit 10k steps."""
    inputs = np.array([[sleep_hr, water_glasses, bench_kg]])
    pred = clf.predict(inputs)[0]
    proba = clf.predict_proba(inputs)[0]
    confidence = round(max(proba) * 100, 1)

    result = {
        "hit_goal": bool(pred),
        "confidence": confidence,
        "recommendation": ""
    }

    if pred == 0 and confidence > 70:
        result["recommendation"] = "Low step count likely. Schedule a walk."
    elif pred == 1 and confidence > 70:
        result["recommendation"] = "High step count likely. Keep it up!"
    else:
        result["recommendation"] = "Borderline day. Stay intentional about movement."

    return result

# Test with scenarios
scenarios = [
    (8.0, 8, 84, "James Omondi"),
    (6.0, 5, 78, "Brian Kamau"),
    (9.0, 9, 87, "Grace Achieng"),
]

for sleep, water, bench, name in scenarios:
    r = predict_day(sleep, water, bench)
    outcome = "Goal hit" if r["hit_goal"] else "Below goal"
    print(f"{name}: {outcome} ({r['confidence']}% confidence)")
    print(f" Recommendation: {r['recommendation']}\n")
