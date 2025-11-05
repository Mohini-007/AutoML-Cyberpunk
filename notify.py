import requests

TOKEN = ""   # <-- replace this with token
CHAT_ID =           # <-- replace this with chat I'd of telegram bot (just number)

def send_notification(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": message}
    r = requests.get(url, params=params)
    print("Status:", r.status_code, "→", r.text)

if __name__ == "__main__":
    send_notification("✅ AutoML Bot connected successfully!")

