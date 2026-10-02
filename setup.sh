#!/bin/bash
# GLDRUBF-Sentry: Первичная установка и настройка
set -e

echo "========================================"
echo "  GLDRUBF-Sentry Setup"
echo "========================================"

# 1. Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Python $PYTHON_VERSION найден"

# 2. Создание виртуального окружения
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Создаю виртуальное окружение..."
    python3 -m venv venv
    echo "✅ Виртуальное окружение создано"
else
    echo ""
    echo "✅ Виртуальное окружение уже существует"
fi

# 3. Активация и установка зависимостей
echo ""
echo "📦 Устанавливаю зависимости..."
# shellcheck disable=SC1091
source venv/bin/activate

pip install --upgrade pip -q
pip install certifi -q

# opensource.tbank.ru (PyPI-индекс T-Invest SDK) использует сертификат от
# «Russian Trusted CA» (МИДцифры), которого нет в стандартном хранилище CA,
# поэтому без доп. корня pip падает с CERTIFICATE_VERIFY_FAILED.
# Собираем объединённый CA-бандл (системные Root + Russian Trusted CA из certs/)
# и указываем его pip — проверку TLS НЕ отключаем.
CA_BUNDLE="$(python tools/pip_ca_bundle.py "$(pwd)/venv/pip-ca-bundle.pem")"
PIP_CERT="$CA_BUNDLE" pip install -r requirements.txt \
    --extra-index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple -q
echo "✅ Зависимости установлены"

# Инвест--API T-Bank (invest-public-api.tbank.ru) тоже использует сертификат от
# Russian Trusted CA — для requests/grpcio задаём то же доверяемое хранилище.
cat >> venv/bin/activate <<ACTIVATE

# Добавлено setup.sh: доверять Russian Trusted CA (сертификаты *.tbank.ru)
export REQUESTS_CA_BUNDLE="${CA_BUNDLE}"
export SSL_CERT_FILE="${CA_BUNDLE}"
ACTIVATE
echo "✅ venv/bin/activate дополнен переменными REQUESTS_CA_BUNDLE / SSL_CERT_FILE"

# 4. Проверка token.env
if [ ! -f "token.env" ]; then
    echo ""
    echo "⚠️  Файл token.env не найден!"
    echo "   Создайте файл token.env с содержимым:"
    echo "   T_SANDAPI=ваш_токен_T_Invest_API"
else
    if grep -q "T_SANDAPI=" token.env && [ -s token.env ]; then
        echo "✅ token.env найден и содержит T_SANDAPI"
    else
        echo "⚠️  token.env существует, но T_SANDAPI не задан"
    fi
fi

# 5. Итоговое сообщение
echo ""
echo "========================================"
echo "  ✅ Установка завершена!"
echo "========================================"
echo ""
echo "Запуск бота:"
echo "  source venv/bin/activate"
echo "  export \$(grep -v '^#' token.env | xargs)"
echo "  python -m src.main"
echo ""
echo "Или одной командой:"
echo "  source venv/bin/activate && export \$(grep -v '^#' token.env | xargs) && python -m src.main"
echo ""
