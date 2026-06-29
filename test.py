import sys
import os
import platform
import traceback

print("=" * 60)
print("Python executable:")
print(sys.executable)

print("\nPython version:")
print(sys.version)

print("\nPlatform:")
print(platform.platform())

print("\nConda environment:")
print(os.environ.get("CONDA_DEFAULT_ENV", "Not in conda env"))

print("\nPATH first 10:")
for p in os.environ.get("PATH", "").split(os.pathsep)[:10]:
    print(p)

print("\nTrying to import torch...")
try:
    import torch
    print("torch import: OK")
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu name:", torch.cuda.get_device_name(0))
except Exception:
    print("torch import: FAILED")
    traceback.print_exc()

print("\nTrying to import basic libraries...")
libs = ["numpy", "pandas", "mne", "sklearn", "timm"]
for lib in libs:
    try:
        module = __import__(lib)
        print(f"{lib}: OK")
    except Exception:
        print(f"{lib}: FAILED")
        traceback.print_exc()

print("=" * 60)