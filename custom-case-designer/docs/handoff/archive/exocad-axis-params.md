# Опись параметров оси exocad на ПК

## Обратная передача
Исполнитель: Codex · Ветка: follower/pc-checks · Коммит: этот коммит (история файла).
Сделано: а) прочитаны 149 XML: 6 defaultparameters, 137 библиотечных articulatorparameters,
ещё 6 config/articulatorparameters; 44 различных файла по SHA-256, 17 901 значений/атрибутов.
Отклонения от контракта: нет; программы и библиотеки не изменялись, exocad не запускался.
Проверки: XML разобраны, каждый файл повторно сверен по SHA-256; полный реестр ниже.
Карте: опись параметров для независимого положения головок выполнена; нужен опыт со вставками Беннетта.
Вопросы ведущему: проверить RotationAxis по одной стороне; отдельным опытом — механику ImmediateSideShiftRotates.

Срез файлов: 13.10.2026. Проверены установленные DentalCADApp 3.3/3.2, ImplantPlanning,
OrthoCADApp и найденные архивные/дублирующие папки под `C:\Exo`.
Каталоги INJECTOR и .ECFG исключены. Других корней установки в эту опись не включено.
Существующее описание из `origin/lead/exocad-articulator-export:docs/exocad.md`
(«Артикуляторы») здесь не пересказывается. Ниже — новые находки и точный реестр.

## Полный реестр

- [sources.csv](exocad-axis-params/sources.csv): каждый найденный файл, путь относительно
  `C:\Exo`, SHA-256 и идентификатор профиля P001…P044. Одинаковые файлы объединены
  только по полному совпадению байтов, архивные варианты сохранены отдельно.
- [values.csv](exocad-axis-params/values.csv): **все** конечные XML-значения и атрибуты
  каждого профиля, включая все параметры defaultparameters (не только артикулятор).
  Порядковые индексы сохраняют повторяющиеся элементы, в том числе дубли флагов.
  Для defaultparameters найдите `Parameter[N]/Type`, затем Value/MinValue/MaxValue
  с тем же N. Для библиотеки — `<Имя>Param/{Type,Value,MinValue,MaxValue}`.
  Пустое значение сохранено пустым; NaN не заменён нулём. Отсутствие узла не означает false/0.
- [ranges.csv](exocad-axis-params/ranges.csv): плоский индекс всех блоков параметров
  с Type, Value, MinValue, MaxValue, включая вложенные переопределения протезов.

## Новое относительно прежнего описания

| Параметры | Точные находки | Значение для поиска независимых головок |
|---|---|---|
| `RotationPointShift` | Один скаляр: обычно 1.2; GAMMA — 2.5; Ivoclar обычный — 0, Adjustable — −4; KaVo 7 — 0 | Нет отдельных Left/Right. Единицы, направление и объект воздействия XML не объясняет; считать его перемещением головок пока нельзя |
| `ImmediateSideShiftLeftPivotPoint` / `RightPivotPoint` | Ivoclar 300 и Adjustable: (−1,15,0)/(1,15,0); KaVo 7: (0,0,0) с обеих сторон | Реальные отдельные трёхмерные точки по сторонам, но по именам относятся к механизму бокового сдвига, не к положению головок в покое |
| `ImmediateSideShiftRotates`, `ImmediateSideShiftMaxRotationAngle` | Ivoclar: true и 6.0; KaVo 7: false и 0 | Кандидат для отдельного опыта: сдвиг реализуется поворотом вокруг указанных pivot; это предположение по группе полей, не проверка механики |
| `ImmediateSideshiftAlsoShiftsBennetInsert` | Ivoclar: true; KaVo 7: false | Связь бокового сдвига со вставкой Беннетта явно вынесена в настройку |
| `CondylarAngleDoesNotRotateBennetInsert` | Ivoclar: true; KaVo 7: false | Отдельное управление связью угла дорожки и вставки; написание Bennet с одной t сохранено |
| `FrontplateAdjustablePivotPointOffsetLeft/Right` | bio-art Adjustable: (2.75,0,0.64)/(−2.75,0,0.64); KaVo 7: (0,0,3) обе стороны | Отдельные векторы есть, но это резцовый столик. Не основание считать их управлением шарнирной осью |
| `FrontplateAdjustablePivotPoint` | bio-art Adjustable z=−0.59; Ivoclar Adjustable/Artex Adjustable/exoskull/Harman z=−2; SAM 2P z=−3; KaVo 7 z=3 | Общая точка столика; x=y=0 во всех перечисленных вариантах |
| `ExecuteRetrusionWithZeroBennettAngle` | SAM 2P/Panadent и ряд других — true; SAM NEO/KaVo 5B — false | Изменяет условие возвратного движения, а не независимую координату головки |
| `RotationAngleFor2DImageBasedRegistration` | Во всех найденных явных значениях — 10.0 | Относится к регистрации по 2D, Left/Right нет |
| `IncisalNeedleRotationFactorUnitToDegree` | 0.49 (AD2), 0.666667 (Denar), 0.44 (Ivoclar Adjustable/Artex Adjustable/exoskull/Harman Occludator), 1.0 (Panadent PCH) | У штифта единицы явно переводятся в градусы отдельным полем; для RotationAxis такого коэффициента не найдено |

Для векторов выше min/max в XML отсутствуют. Числа — буквально из файлов;
единицы и трактовку pivot нужно подтвердить опытом или документацией поставщика.
`ArticulatorPosition` и `HeightUpperArticulatorPart` остаются общими параметрами;
отдельных координат `CondylarHeadPositionLeft/Right` или матриц положения головок
в прочитанных файлах не обнаружено. Это ограничение наблюдаемых конфигураций,
а не доказательство отсутствия такой возможности у exocad.

## RotationAxis — отдельно

1. Во всех 6 defaultparameters присутствуют обе стороны: `Value=NaN`,
   `MinValue=0`, `MaxValue=6`. Текстов единиц/описания в этих блоках нет.
2. Во всех 137 библиотечных articulatorparameters `CollapseRotationAxis=true`.
   Явные блоки `RotationAxisLeftParam/RightParam` найдены только у установленного
   Custom Case Designer (P025): обе стороны `Value=0`, диапазон 0…6.
   У Harman OSH Articulator (P029) и Occludator (P030) этих блоков нет.
3. Других параметров с RotationAxis в именах/Type, а также коэффициента перевода
   его единиц в градусы в собранном реестре нет. Переопределений RotationAxis
   внутри FullDenture* также нет. NaN в defaultparameters не доказывает, каким
   будет эффективное значение после загрузки конкретной библиотеки.
4. XML не разрешает спор «мм или градусы», не доказывает сдвиг головки и не
   исключает поворот дорожки. Следующий опыт ведущего: неизменные модели и
   парные вставки ССП/Беннетта, Left=3/Right=0, затем наоборот и оба=0;
   отдельно измерить центры головок, точки оси и дорожки в покое и открывании.

## Сторонние библиотеки

Harman OSH Articulator: межмыщелковое 100; RotationPointShift=1.2;
ССП слева/справа Value=35, диапазон 0…75; Беннетт Value=0, диапазон 0…30;
ретрузия Value=0, диапазон 0…2; явных ImmediateSideshift*Param нет;
RotationAxis скрыт, явных значений нет. Профиль P029 одинаков в найденных копиях.

Harman OSH Occludator: межмыщелковое 30; RotationPointShift=1.2;
ССП Value=45, диапазон −20…89; ретрузия Value=0, диапазон 0…10;
боковой сдвиг Value=0, диапазон 0…1 по обеим сторонам. Профиль P030.
Название Occludator существенно: переносить его параметры на артикулятор без
опыта нельзя. Точные значения остальных параметров и всех библиотек — в реестре.
