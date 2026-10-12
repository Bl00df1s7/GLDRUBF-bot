# Как опубликовать содержимое PR #39

Готовый коммит в репозитории `/workspace`:
- ветка: `qwen-code-7ef9d2e6-3be1-46a5-abcf-bd47341a9bcd`
- коммит: `b4dc9d5` (parent = `f515f2d`, актуальный `main`)
- message: `chore: untrack committed .pyc caches and refresh stale G-5 comment after PR #37`

 Sandbox без учётных данных GitHub (remote не настроен), поэтому push выполняется у вас локально.

## Вариант A — обновить существующий PR #39 (рекомендуемый)

Ваша PR-ветка на GitHub называется `audit-pr-37-open-orders-a9bcd`. Она была перезаписана amend-ом,
поэтому нужен force-with-lease:

```bash
cd <локальная_копия_GLDRUBF-bot>
git fetch origin
git checkout -B audit-pr-37-open-orders-a9bcd b4dc9d5   # или примените patch из варианта B
git push --force-with-lease origin audit-pr-37-open-orders-a9bcd
```

PR #39 подхватит новый коммит автоматически. Тело PR — из `PR_BODY.md`.

## Вариант B — применить патч к своей копии

```bash
git am pr39.patch          # полный коммит (author/message/diff)
# или, если хотите свой коммит:
git apply pr39.diff        # только изменения в рабочем дереве
```

Проверка после применения:

```bash
git diff --stat HEAD~1     # должно быть: 51 files changed, 11 insertions(+), 5 deletions(-)
git ls-files '*.pyc'       # пусто
git diff origin/main -- .gitignore   # пусто
```

## Вариант C — gh CLI (если предпочитаете заново загрузить файлы)

Бинарные `.pyc` в `pr39.diff` представлены как «Binary files differ» без содержимого — для
ручного копирования файлов используйте именно `git am`/`format-patch` (вариант B) или push
варианта A; они содержат полные blob-данные удаления.

## Что НЕ делать

- Не мержить PR без вашего решения.
- Не переписывать историю `main`.
- Не удалять `.pyc` с диска (использован только `git rm --cached`).

## Файлы в этом каталоге

| Файл | Назначение |
|---|---|
| `pr39.patch` | Полный коммит `b4dc9d5` в формате `git format-patch` (применяется через `git am`) |
| `pr39.diff` | Diff `f515f2d..b4dc9d5` (текстовая часть; бинарные `.pyc` показаны как удалённые) |
| `changed_files.txt` | Список всех 51 изменённых файлов со статусами (D/M) |
| `diff_stat.txt` | Итоговая статистика diff |
| `c5_runtime_comment.diff` | Только ханк комментария G-5 в `src/c5_runtime.py` (для ревью) |
| `PR_BODY.md` | Готовое тело/описание PR #39 |
| `PUBLISH_HOWTO.md` | Эта инструкция |
