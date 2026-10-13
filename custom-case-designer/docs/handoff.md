# Передача работы

Порядок работы — `WORK_SPLIT.md`. Здесь — только доска; каждая передача — в своём файле
`docs/handoff/<задача>.md` по шаблону (`WORK_SPLIT.md`, «Передача: шаблон»), принятые — в
`docs/handoff/archive/`. Доску ведёт ведущий.

## Текущее состояние

Обновлено 13.10.2026. Общая ветка: `claude/lucid-euler-8zyldi`.

| Задача | Исполнитель | Ветка | Занятые файлы | Состояние | Передача |
|---|---|---|---|---|---|
| 1. Выгрузка «Для exocad с артикулятором» — интерфейс | Codex | `follower/exocad-export-ui` | `app/session.py`, `app/server.py`, `app/web/app.js`, `app/web/style.css`, `docs/exocad.md` (раздел выгрузки) | **к работе — первая** | `docs/handoff/exocad-articulator-export.md` |
| 2. Плоскость монтажа — интерфейс и настройка (7 плоскостей, по умолчанию — «эстетическая через мыщелки») | Codex | `follower/mount-plane-ui` | те же, после №1 | очередь Codex №2 | `docs/handoff/mount-plane.md` |
| 3. Сустав отдельно: показ головок и скатов, заметка замены точки ALI | Codex | `follower/tmj-segments-ui` | те же, после №2 | очередь Codex №3 | `docs/handoff/tmj-segments.md` |
| 4. Проверка на архиве КТ: головки против ALI, скаты, ССП | Codex | — (на ПК, вне репозитория) | только `docs/handoff/tmj-archive-check.md` | можно параллельно с №1 | `docs/handoff/tmj-archive-check.md` |
| «Кривая» ось в exocad: шарнир точно + эстетические виды | Claude | `lead/exocad-aesthetic-views` | — | принято в общую ветку (`829fbb1`): «эстетическая через мыщелки» + «Вид Smile Design» из `.treatmentImages`; остаток к эстетике 0,9° (кейс 3) | `docs/exocad.md`, «Эстетический вид» |
| Сустав отдельно: головки и скаты (ядро) | Claude | `lead/tmj-segments` | `landmarks.py`, `guidance.py`, `condyles.py`, `exocad_webview.py` | каркас готов (идея владельца): головки по связности сетки, ALI > 4 мм → головка, скаты, ССП 5–75° | `docs/handoff/tmj-segments.md` |
| Каркас «Исправить прикус» (отказ без контактной опоры, предел поправки) | Claude | `lead/bite-correction` | `bite.py` | следующая у ведущего | `docs/handoff/archive/bite-archive-check.md` (дефект) |
| Разделение `session.py`, `server.py`, `app.js` по шагам | Claude | `lead/split-steps` | — | отложено до приёмки №1–3 (одни файлы) | — |
| Опись параметров exocad; порог перекоса; прикус на архиве | Codex | `follower/pc-checks` | — | принято (`bb6841b`, `424a79e`, `ba6c208`) | `docs/handoff/archive/` |
| Ручная проверка и поправка мыщелков | Codex, Claude | — | — | принято (`bc0476b`) | `docs/handoff/archive/condyle-adjustment.md` |

Очередь Codex (интерфейс — по одной, одни файлы): №1 выгрузка для exocad → №2 плоскость монтажа → №3
суставы; №4 (проверка на архиве) — параллельно, файлы не пересекаются. Каждая — своя ветка от последнего
коммита общей ветки, перед началом — `git fetch`.

Очередь ведущего: каркас «Исправить прикус» → приёмка №1–4 → разделение файлов интерфейса по шагам →
разбор входящих и чистка карты → ядро суставного пути в 3D.

Вопросы владельцу: публикация моделей (`models-v1`, `landmarks-v1` — на GitHub их нет, скачивание из
программы не работает; для `landmarks-v1` таблица хешей в `model_store.py` должна совпасть с
публикуемыми файлами).
