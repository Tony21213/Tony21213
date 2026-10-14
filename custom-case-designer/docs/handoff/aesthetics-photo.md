# «Эстетика», часть 1: шаг «Эстетика» — фото с улыбкой и с ретрактором

Исполнитель: Claude · Ветка: lead/aesthetics · Каркас в общей ветке: fad455c · Состояние: каркас готов

## Ведомому

Цель: шаг «Эстетика» после «Сканы»: фото анфас с улыбкой и с ретрактором, каждое совмещено со сканом по
зубам; сцена выравнивается по фото; оба фото — в Smile Design exocad со своими камерами; размеры
передних зубов по лицу в мм.

Сделать:
1. Сессия: `photos` — `{"smile"|"retracted": {"photo": aesthetic.load_photo(путь), "pairs": [(пиксель,
   точка скана)], "camera": None, "horizon_deg": 0, "midline_px": []}}`; `aesthetic_add_photo(kind,
   path)`, `aesthetic_pair(kind, px, point)` / удалить пару, `aesthetic_fit(kind)` —
   `aesthetic.fit_camera(px, точки, size, база, photo.focal_px)` (база — система монтажа
   `jaw.anatomy.frame`; 4+ пар, без EXIF — 6+), ошибка в пикселях и `mirrored` — в состоянии;
   `face_hints` — подсказки лица (модель MediaPipe — как модели ориентиров). Фото — в `.ccdcase` путём
   (как сканы), пары и горизонт — в meta. Проверка: тесты сессии на синтетике (как
   `tests/test_exocad_photo_views.py::camera`).
2. Монтаж по фото: в шаге «Артикулятор» при плоскости «эстетическая» / «эстетическая через мыщелки» и
   совмещённом фото с улыбкой — `jaw.mount(..., photo_aesthetic=lambda base:
   aesthetic.aesthetic_frame(base, camera, horizon_deg, midline_px, incisal))`; без фото — по костям
   лица, как сейчас. Проверка: тест — заметка «по эстетике (фото».
3. Выгрузка для exocad: `ArticulatorCase.photos` — `[{"kind", "image", "camera"}]` из сессии
   (`case_from_session`): exocad открывает фото с улыбкой по «Вид Smile Design». Проверка: HTTP-тест —
   роль «эстетический вид» с файлами `-smile.jpg`, `-retracted.jpg`.
4. Интерфейс: загрузка двух фото, пары точек «фото — скан» щелчками (на фото и на скане в 3D), наложение
   скана на фото для проверки (проекция `camera.project`), горизонт и средняя линия с ручной правкой,
   «Размеры зубов по лицу» — `aesthetic.face_widths_mm` + `tooth_guide` (с пояснением «ориентир»). В
   инкогнито лицо на фото размывать; путь к фото в журнале ошибок скрывать (`errorlog.private`).
5. Браузер на фантоме: синтетическое фото (проекция скана), пары, камера, монтаж, выгрузка.

Можно менять: `app/session.py`, `app/server.py`, `app/web/app.js`, `app/web/style.css`, новые тесты.
Не трогать: `aesthetic.py`, `exocad_views.py`, `jawcase.py`, `exocad_articulator_export.py`.

Готово, когда: новые тесты проходят; полный набор проходит; `git diff --check` чисто; на фантоме фото
совмещается и выгружается в набор Smile Design.

## Контракт

Источник истины — `aesthetic.fit_camera`, `aesthetic_frame`, `face_widths_mm`, `tooth_guide`;
`exocad_views.photo_view`, `write_photo_images`; `JawCase.mount(photo_aesthetic=...)`; тесты
`tests/test_exocad_photo_views.py`, `tests/test_aesthetic.py`, `tests/test_mount_planes.py::
test_aesthetic_hinge_by_photo_without_face_bones`, `tests/test_exocad_articulator_export.py::
test_patient_photos_go_to_smile_design_by_camera`.
Сверх кода:
- Каждое фото совмещается со сканом отдельно, по своим зубам: фото между собой согласованы через 3D
  (решение владельца 13.10); одно фото на оба случая — запасной вариант.
- В exocad: фото с улыбкой — кнопка «Вид Smile Design»; ретрактор — «показать/скрыть». Проверить в
  exocad, что каждый снимок встаёт по своей камере (ведущий, после интерфейса).
- Размеры зубов — литературные соотношения (ориентир), глубина точек лица — средняя пропорция профиля
  (`FACE_DEPTH_MM`); точнее — со сканом лица или кожей по КТ (позже).
- Данные пациента: фото лица узнаваемо — только вне репозитория; в отчётах — номер кейса без имени.
- Часть 2 — цефалометрия (порядок работ, п. 10).

## Обратная передача
