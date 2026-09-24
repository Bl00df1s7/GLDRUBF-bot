#!/usr/bin/env python3
"""Собирает объединённый CA-бандл для pip/requests.

Проблема: индекс https://opensource.tbank.ru (T-Invest SDK) выдаёт сертификат,
выпущенный «Russian Trusted Sub CA» (МИДцифры). Его корень отсутствует и в
системном хранилище CA (ca-certificates), и в пакете certifi, поэтому pip падает с
`SSL: CERTIFICATE_VERIFY_FAILED ... unable to get local issuer certificate`.

Решение без отключения проверки: берём системный набор Root-сертификатов
(или certifi — что найдётся), добавляем к нему сертификаты Russian Trusted
(sub CA + Root CA из каталога certs/) и пишем единый bundle. Дальше его можно
передать pip через `--cert <bundle>` или переменную окружения PIP_CERT, а также
использовать как REQUESTS_CA_BUNDLE для runtime.

Использование:
    python tools/pip_ca_bundle.py [путь_для_записи]   # по умолчанию /tmp/pip-ca-bundle.pem
"""
from __future__ import annotations

import os
import sys

try:
    import certifi
except ImportError:  # certifi не обязателен
    certifi = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUSSIAN_TRUSTED_BUNDLE = os.path.join(REPO_ROOT, "certs", "russian-trusted-ca-bundle.pem")

# Кандидаты системного хранилища Root-сертификатов (в порядке приоритета)
SYSTEM_CA_CANDIDATES = [
    "/etc/ssl/certs/ca-certificates.crt",  # Debian/Ubuntu (GitHub Actions ubuntu-latest)
    "/etc/pki/tls/certs/ca-bundle.crt",    # RHEL/Fedora/CentOS
    "/etc/ssl/cert.pem",                   # Alpine/macOS
]


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def build_bundle() -> bytes:
    parts: list[bytes] = []

    system_ca = next((p for p in SYSTEM_CA_CANDIDATES if os.path.isfile(p)), None)
    if system_ca is not None:
        parts.append(read_bytes(system_ca))
    elif certifi is not None:
        parts.append(read_bytes(certifi.where()))
    else:
        raise SystemExit(
            "Не найдено ни системное хранилище CA, ни пакет certifi — "
            "не из чего строить доверяемый бандл."
        )

    if os.path.isfile(RUSSIAN_TRUSTED_BUNDLE):
        parts.append(b"\n" + read_bytes(RUSSIAN_TRUSTED_BUNDLE))
    else:
        print(
            f"Внимание: не найден {RUSSIAN_TRUSTED_BUNDLE} — "
            "сертификаты Russian Trusted CA не будут добавлены.",
            file=sys.stderr,
        )

    return b"".join(parts)


def main(argv: list[str]) -> int:
    out_path = argv[1] if len(argv) > 1 else "/tmp/pip-ca-bundle.pem"
    data = build_bundle()
    with open(out_path, "wb") as fh:
        fh.write(data)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
