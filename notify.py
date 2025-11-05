import requests

TOKEN = "8467883257:AAEFn4nYxKdGdpeBSKLjwVuHHP51Eb_oED8"   # <-- replace this
CHAT_ID = 5361232040          # <-- replace this (just number)

def send_notification(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": message}
    r = requests.get(url, params=params)
    print("Status:", r.status_code, "→", r.text)

if __name__ == "__main__":
    send_notification("✅ AutoML Bot connected successfully!")
