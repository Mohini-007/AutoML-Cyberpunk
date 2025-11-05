import pandas as pd
import joblib
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# === Telegram Notify ===
TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = YOUR_CHAT_ID

def notify(msg):
    requests.get(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                 params={"chat_id": CHAT_ID, "text": msg})

# === Load Data ===
df = pd.read_csv("dataset.csv")   # ensure this exists in repo
target_col = "price"              # adjust if needed

X = df.drop(columns=[target_col])
y = df[target_col]

# Preprocess
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), numeric_features)]
)

model = RandomForestRegressor(n_estimators=300, random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, preds)

# Save Model
joblib.dump(pipeline, "best_model_pipeline.joblib")

# Send Telegram Notification 🚀
notify(f"✅ Auto Retrain Complete!\n🧠 Model: RandomForest\n📉 MAE: {mae:.2f}")
print(f"Retraining Done. MAE: {mae:.2f}")
