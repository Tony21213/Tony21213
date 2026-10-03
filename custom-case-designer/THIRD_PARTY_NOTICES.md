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

## Сегментация

Модели сегментации подключаются только с лицензиями, разрешающими
использование в приложении, с указанием авторства здесь и в окне «О
программе». Найденные модели, их лицензии, авторы и ограничения — в
[docs/models.md](docs/models.md). Основные:

- **TotalSegmentator** (задачи `teeth`, `craniofacial_structures`,
  `head_glands_cavities`) — Apache-2.0. Wasserthal J. et al.,
  "TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT
  Images", *Radiology: Artificial Intelligence* (2023).
  https://github.com/wasserth/TotalSegmentator
  - модель `teeth` обучена на наборе данных **ToothFairy3** (CC BY-NC-SA):
    Bolelli F. et al., "Segmenting Maxillofacial Structures in CBCT
    Volumes", CVPR 2025.
- **ToothSeg** — веса CC BY 4.0, код Apache-2.0. Isensee F., van Nistelrooij N.,
  Krämer L., Vinayahalingam S. et al. (2025). https://github.com/MIC-DKFZ/ToothSeg,
  https://zenodo.org/records/14893540. Обучено на **ToothFairy2** (CC BY-SA).
- **DentalSegmentator** — CC BY 4.0. Dot G. et al., "DentalSegmentator:
  robust open source deep learning-based CT and CBCT image segmentation",
  *Journal of Dentistry* (2024). https://doi.org/10.5281/zenodo.10829675
- **nnU-Net** (архитектура и обучение всех моделей выше) — Apache-2.0.
  Isensee F. et al., *Nature Methods* 18, 203–211 (2021).

Код и веса OdentAI не используются: для них не указана лицензия,
разрешающая переиспользование.
