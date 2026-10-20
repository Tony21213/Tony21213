# Сторонние компоненты и лицензии

KStom Case Designer использует следующие открытые библиотеки. Их лицензии
требуют сохранять уведомления об авторских правах — они перечислены здесь и
поставляются вместе с приложением.

| Компонент | Для чего | Лицензия |
|---|---|---|
| [NumPy](https://numpy.org) | вычисления | BSD-3-Clause |
| [SciPy](https://scipy.org) | интерполяция, поиск ближайших точек | BSD-3-Clause |
| [scikit-image](https://scikit-image.org) | пороги Оцу, marching cubes | BSD-3-Clause |
| [SimpleITK](https://simpleitk.org) | чтение DICOM, NIfTI, MHA, NRRD | Apache-2.0 |
| [trimesh](https://trimesh.org) | чтение и запись STL/PLY/OBJ, сечения сеток | MIT |
| [ONNX Runtime](https://onnxruntime.ai) | запуск моделей сегментации (CPU, DirectML) | MIT |
| [Pillow](https://python-pillow.org) | картинки срезов | MIT-CMU (HPND) |
| [three.js](https://threejs.org) 0.170 | 3D в интерфейсе (`casedesigner/app/web/vendor/three`, лицензия рядом) | MIT |
| [pywebview](https://pywebview.flowrl.com) | окно приложения | BSD-3-Clause |
| [MediaPipe](https://ai.google.dev/edge/mediapipe) и модель Face Landmarker | точки лица на фото (эстетическая система, необязательно) | Apache-2.0 |
| [Matplotlib](https://matplotlib.org) | картинка анализа движений (`motion -o`, необязательно) | Matplotlib License (PSF-based) |
| [h5py](https://www.h5py.org) | записи движений в HDF5 (необязательно) | BSD-3-Clause |
| [PyInstaller](https://pyinstaller.org) | сборка exe (только при сборке) | GPL-2.0 с исключением для сборок |

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

## Фото и сканы

- **SAM 2.1** (Segment Anything 2.1, Hiera-Tiny) — Apache-2.0. Ravi N. et al., "SAM 2: Segment Anything in
  Images and Videos" (Meta, 2024), https://github.com/facebookresearch/sam2. ONNX — vietanhdev,
  https://huggingface.co/vietanhdev/segment-anything-2.1-onnx-models.
- **MoGe-2** (ViT-S) — MIT. Wang R. et al., "MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp
  Details" (Microsoft, 2025), https://github.com/microsoft/MoGe; DINOv2 внутри — Apache-2.0.
- **GeoCalib** — код Apache-2.0, веса CC BY 4.0. Veicht A., Sarlin P.-E., Lindenberger P., Pollefeys M.,
  "GeoCalib: Learning Single-image Calibration with Geometric Optimization", ECCV 2024,
  https://github.com/cvg/GeoCalib.
- **DentalModelSeg** — Apache-2.0. DCBIA-OrthoLab (University of North Carolina at Chapel Hill, University of
  Michigan): Leclercq M., Prieto J. C., Cevidanes L., Styner M. et al.,
  https://github.com/DCBIA-OrthoLab/SlicerDentalModelSeg (веса — релиз v3.0).

## Автоматические ориентиры

- **ALI-CBCT** — сети поиска цефалометрических точек на КЛКТ (переведены в
  ONNX, веса хранятся в float16). Gillot M., Miranda F., Baquero B. et al.,
  "Automatic landmark identification in cone-beam computed tomography",
  *Orthodontics & Craniofacial Research* (2023),
  https://doi.org/10.1111/ocr.12642. Веса — из релиза `v0.1-v2.0_models`
  репозитория DCBIA-OrthoLab/SlicerAutomatedDentalTools, распространяемого
  под лицензией 3D Slicer (BSD-подобная):
  https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools. Алгоритм
  поиска (агент, шаги, итоговые пробы) повторён в
  `casedesigner/auto_landmarks.py` без исходного кода. В выпуске `landmarks-v2` (14.10.2026) —
  ещё точки цефалометрии A, B, Pog, Gn, Me, Go и правых центральных резцов
  (архивы `Lower_Bones_1/2`, `Upper_Bones_v2`, `Upper_Right_Teeth_v2`,
  `Lower_Right_Teeth` того же релиза).
- **MONAI** (архитектура DenseNet этих сетей; нужна только при переводе
  весов, `tools/prepare_landmarks.py`) — Apache-2.0.

Код и веса OdentAI не используются: для них не указана лицензия,
разрешающая переиспользование.
