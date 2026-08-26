import os
import sys
import subprocess
import urllib.request
import zipfile

def print_step(text):
    print(f"\n[+] {text}")

# 1. Crear estructura de carpetas
print_step("Creando estructura de carpetas del proyecto...")
folders = [
    "rangelyze-ecosystem/sandbox/core",
    "rangelyze-ecosystem/sandbox/weights",
    "rangelyze-ecosystem/datasets",
    "rangelyze-ecosystem/rangelyze_core/models",
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"  ✓ Carpeta lista: {folder}")

# 2. Instalar librerías necesarias
print_step("Instalando librerías requeridas vía PIP...")
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
    print("  ✓ Todas las librerías fueron instaladas correctamente.")
except Exception as e:
    print(f"  ❌ Error instalando librerías: {e}")

# 3. Descargar e inicializar pesos en caché local
print_step("Descargando pesos de los modelos a la caché local...")

try:
    print("  -> Descargando YOLOv8n (Detector)...")
    from ultralytics import YOLO
    _ = YOLO('yolov8n.pt')
    print("  ✓ YOLOv8n guardado en caché local.")
except Exception as e:
    print(f"  ❌ Error descargando YOLOv8n: {e}")

try:
    print("  -> Descargando MobileNetV3 Small (Clasificador)...")
    import torchvision.models as models
    _ = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    print("  ✓ MobileNetV3 guardado en caché local.")
except Exception as e:
    print(f"  ❌ Error descargando MobileNetV3: {e}")

# 4. Descargar Dataset BCCD
print_step("Descargando el Dataset BCCD en ZIP...")
bccd_dir = "rangelyze-ecosystem/datasets/BCCD"
zip_path = "rangelyze-ecosystem/datasets/bccd.zip"
url = "https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip"

try:
    if not os.path.exists(zip_path) and not os.path.exists(bccd_dir):
        print("  -> Descargando repositorio desde GitHub...")
        urllib.request.urlretrieve(url, zip_path)
        print("  -> Descomprimiendo archivos...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("rangelyze-ecosystem/datasets/")
        os.rename("rangelyze-ecosystem/datasets/BCCD_Dataset-master", bccd_dir)
        os.remove(zip_path)
        print(f"  ✓ Dataset BCCD listo en: {bccd_dir}")
    else:
        print("  ✓ El dataset BCCD ya existe en disco.")
except Exception as e:
    print(f"  ❌ Error al descargar el dataset BCCD: {e}")
    print("  (Si falla, puedes clonarlo manualmente con: git clone https://github.com/Shenggan/BCCD_Dataset.git)")

print("\n" + "="*50)
print("🚀 ENTORNO OFFLINE PREPARADO CON ÉXITO")
print("Ya puedes desconectarte a la red y trabajar 100% en local.")
print("="*50)