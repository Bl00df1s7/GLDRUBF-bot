import os
import requests
import time

BOT_TOKEN = os.environ["BOT_TOKEN"]
GITHUB_TOKEN = os.environ["GH_TRIGGER_TOKEN"]

OWNER = os.environ["GITHUB_OWNER"]
REPO = os.environ["GITHUB_REPO"]
WORKFLOW = "gold_strategy.yml"

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"


def send_message(chat_id, text):
    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text,
        },
        timeout=10,
    )


def trigger_workflow():
    url = (
        f"https://api.github.com/repos/"
        f"{OWNER}/{REPO}/actions/workflows/{WORKFLOW}/dispatches"
    )

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        },
        json={
            "ref": "main",
        },
        timeout=10,
    )

    response.raise_for_status()


def main():
    offset = None

    print("🤖 Telegram bot started")

    while True:
        response = requests.get(
            f"{TELEGRAM_API}/getUpdates",
            params={
                "timeout": 30,
                "offset": offset,
            },
            timeout=35,
        )

        response.raise_for_status()

        for update in response.json().get("result", []):
            offset = update["update_id"] + 1

            message = update.get("message")
            if not message:
                continue

            chat_id = message["chat"]["id"]
            text = message.get("text", "").strip()

            if text == "/signal":
                send_message(
                    chat_id,
                    "⏳ Запускаю расчёт сигнала..."
                )

                try:
                    trigger_workflow()

                    send_message(
                        chat_id,
                        "✅ Расчёт запущен.\n"
                        "Результат придёт сюда после выполнения."
                    )

                except Exception as e:
                    print(f"ERROR: {e}")

                    send_message(
                        chat_id,
                        "❌ Не удалось запустить расчёт."
                    )


if __name__ == "__main__":
    main()
