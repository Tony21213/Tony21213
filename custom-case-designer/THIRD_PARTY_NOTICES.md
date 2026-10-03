# Сторонние компоненты и лицензии

Custom Case Designer использует следующие открытые библиотеки. Их лицензии
требуют сохранять уведомления об авторских правах — они перечислены здесь и
поставляются вместе с приложением.

| Компонент | Для чего | Лицензия |
|---|---|---|
| [NumPy](https://numpy.org) | вычисления | BSD-3-Clause |
| [SciPy](https://scipy.org) | интерполяция, поиск ближайших точек | BSD-3-Clause |
| [scikit-image](https://scikit-image.org) | пороги Оцу, marching cubes | BSD-3-Clause |
| [SimpleITK](https://simpleitk.org) | чтение DICOM, NIfTI, MHA, NRRD | Apache-2.0 |
| [trimesh](https://trimesh.org) | чтение и запись STL/PLY/OBJ | MIT |

## Сегментация (планируется)

Модели сегментации будут подключаться только с открытыми лицензиями,
разрешающими использование в приложении, с указанием авторства здесь и в
окне «О программе». Кандидаты:

- **DentalSegmentator** — CC BY 4.0. Dot G. et al., "DentalSegmentator:
  robust open source deep learning-based CT and CBCT image segmentation",
  *Journal of Dentistry* (2024). https://doi.org/10.5281/zenodo.10829675
- **nnU-Net** (метод обучения) — Apache-2.0. Isensee F. et al., *Nature
  Methods* 18, 203–211 (2021).
- Открытые наборы данных для дообучения: **ToothFairy2** (CC BY-SA 4.0),
  **DentVoxel** (CC BY 4.0) — с соблюдением условий этих лицензий.

Код и веса OdentAI не используются: для них не указана лицензия,
разрешающая переиспользование.
