import os
import sys
import subprocess
import urllib.request
import zipfile

def print_step(text):
    print(f"\n[+] {text}")

# 1. Crear estructura limpia (Layout src/)
print_step("Creando arquitectura limpia del proyecto (src layout)...")
folders = [
    ".streamlit",
    "assets/models",
    "src",
    "datasets/bccd_sample",
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"  ✓ Carpeta creada: {folder}")

# Crear archivo __init__.py en src/
init_file = os.path.join("src", "__init__.py")
if not os.path.exists(init_file):
    with open(init_file, "w") as f:
        f.write("# Módulo principal Rangelyze\n")

# 2. Instalar librerías
print_step("Instalando librerías necesarias...")
packages = [
    "ultralytics",
    "torch",
    "torchvision",
    "opencv-python",
    "streamlit",
    "pandas",
    "albumentations",
    "pillow",
    "matplotlib"
]

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    print("  ✓ Librerías instaladas.")
except Exception as e:
    print(f"  ❌ Error instalando librerías: {e}")

# 3. Descargar pesos pre-entrenados
print_step("Descargando pesos a la caché local...")
try:
    from ultralytics import YOLO
    _ = YOLO('yolov8n.pt')
    import torchvision.models as models
    _ = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    print("  ✓ Pesos descargados en caché.")
except Exception as e:
    print(f"  ❌ Error descargando pesos: {e}")

# 4. Descargar Dataset BCCD
print_step("Descargando Dataset BCCD...")
zip_path = "datasets/bccd.zip"
url = "https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip"

try:
    if not os.path.exists("datasets/bccd_sample/BCCD"):
        print("  -> Descargando de GitHub...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("datasets/bccd_sample/")
        os.remove(zip_path)
        print("  ✓ Dataset listo en datasets/bccd_sample/")
    else:
        print("  ✓ El dataset ya existía.")
except Exception as e:
    print(f"  ❌ Error en dataset: {e}")

print("\n🚀 ESTRUCTURA Y ENTORNO CONFIGURADOS CORRECTAMENTE")