# PyInstaller: сборка Custom Case Designer в папку с CustomCaseDesigner.exe.
#   pyinstaller --noconfirm packaging/casedesigner.spec
# Модели сегментации кладутся рядом с exe в папку models (см. build_windows.bat).
import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

root = os.path.abspath(os.path.join(SPECPATH, ".."))
web = os.path.join(root, "casedesigner", "app", "web")

datas = [(web, os.path.join("casedesigner", "app", "web"))]
datas += [(os.path.join(root, "casedesigner", "model_specs"), os.path.join("casedesigner", "model_specs"))]
datas += collect_data_files("trimesh")
datas += collect_data_files("webview")
binaries = collect_dynamic_libs("onnxruntime") + collect_dynamic_libs("SimpleITK")
hiddenimports = collect_submodules("webview") + collect_submodules("skimage.measure") + ["PIL.PngImagePlugin"]

a = Analysis(
    [os.path.join(root, "packaging", "launcher.py")],
    pathex=[root],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    # Только для тестов и необязательных частей ядра (движения, эстетика) — в первую версию не входят.
    excludes=["torch", "nnunetv2", "matplotlib", "pytest", "IPython", "tkinter", "onnx", "h5py", "mediapipe"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="CustomCaseDesigner", console=False,
          icon=None, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="CustomCaseDesigner", upx=False)
