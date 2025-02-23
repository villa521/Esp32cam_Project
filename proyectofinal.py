import os
import numpy as np
import torch  # Importar PyTorch
from supabase import create_client
import tempfile
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import cv2  # Asegúrate de tener instalado opencv-python
from ultralytics import YOLO  # Importar YOLO de ultralytics

# ======================
# Configuración
# ======================
SUPABASE_URL = "https://xshchsisefefyazmgewl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhzaGNoc2lzZWZlZnlhem1nZXdsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg0NTAwNjYsImV4cCI6MjA1NDAyNjA2Nn0.OGxuHQ_ApZpC27APq2GdpXuMtACyTqOkwr-DXzC4lT4"
SOURCE_BUCKET = "objetos"
DESTINATION_BUCKET = "deteccion"
MODEL_PATH = r"c:\Users\Villa5050G\Desktop\TACO-master\TACO-master\runs\detect\train4\weights\best2.pt"  # Ruta al modelo YOLOv5 preentrenado

# Configuración de texto mejorada
FONT_SIZE = 48  # Tamaño grande para mejor visibilidad
TEXT_COLOR = (255, 255, 255)  # Blanco
BACKGROUND_COLOR = (0, 0, 0)  # Fondo negro
TEXT_POSITION = (20, 20)       # Posición inicial
BORDER = 10                    # Espacio alrededor del texto
FONT_PATH = "arialbd.ttf"      # Intenta usar Arial Bold

# ======================
# Inicializar clientes
# ======================
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
model = YOLO(MODEL_PATH)  # Cargar modelo YOLOv5

# ======================
# Funciones auxiliares
# ======================
def descargar_imagen_temporal(nombre_archivo):
    """Descarga una imagen a un archivo temporal"""
    try:
        file_data = supabase.storage.from_(SOURCE_BUCKET).download(nombre_archivo)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(nombre_archivo)[1]) as tmp_file:
            tmp_file.write(file_data)
            return tmp_file.name
    except Exception as e:
        print(f"Error descargando {nombre_archivo}: {str(e)}")
        return None

def procesar_imagen_yolov5(ruta_imagen):
    """Realiza la detección con el modelo YOLOv5"""
    results = model(ruta_imagen)
    results_img = results[0].plot()  # Obtener la imagen con detecciones
    
    # Convertir imagen para PIL
    image_rgb = cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

def get_timestamp():
    """Genera un timestamp para evitar colisiones de nombres"""
    return datetime.now().strftime("%Y%m%d%H%M%S")

def guardar_imagen_temporal(imagen_pil, extension):
    """Guarda la imagen modificada en un archivo temporal"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            formato = "JPEG" if extension.lower() in ['.jpg', '.jpeg'] else "PNG"
            imagen_pil.save(tmp_file, format=formato, quality=95)
            return tmp_file.name
    except Exception as e:
        print(f"Error guardando imagen temporal: {str(e)}")
        return None

def subir_imagen(ruta_local, nombre_destino):
    """Sube la imagen modificada al bucket destino"""
    try:
        with open(ruta_local, 'rb') as f:
            file_data = f.read()
        
        extension = os.path.splitext(nombre_destino)[1].lower()
        mime_type = "image/jpeg" if extension in ['.jpg', '.jpeg'] else "image/png"
        
        supabase.storage.from_(DESTINATION_BUCKET).upload(
            nombre_destino,
            file_data,
            file_options={"content-type": mime_type}
        )
        return True
    except Exception as e:
        print(f"Error subiendo {nombre_destino}: {str(e)}")
        return False

# ======================
# Función principal
# ======================
def procesar_y_mover_imagen():
    """Procesa y mueve las imágenes con detección YOLOv5"""
    try:
        archivos = supabase.storage.from_(SOURCE_BUCKET).list()
        imagenes = [f['name'] for f in archivos if f['name'].lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not imagenes:
            print("No se encontraron imágenes para procesar")
            return

        print(f"Procesando {len(imagenes)} imágenes...")
        
        for img_nombre in imagenes:
            tmp_path = None
            tmp_modified_path = None
            try:
                # Descargar y procesar
                tmp_path = descargar_imagen_temporal(img_nombre)
                if not tmp_path:
                    continue
                
                img_con_etiqueta = procesar_imagen_yolov5(tmp_path)
                
                if img_con_etiqueta:
                    # Generar nombres
                    nombre_base, extension = os.path.splitext(img_nombre)
                    nuevo_nombre = f"deteccion_{nombre_base}_{get_timestamp()}{extension}"
                    
                    # Guardar imagen modificada
                    tmp_modified_path = guardar_imagen_temporal(img_con_etiqueta, extension)
                    
                    # Subir y eliminar original
                    if subir_imagen(tmp_modified_path, nuevo_nombre):
                        supabase.storage.from_(SOURCE_BUCKET).remove([img_nombre])
                        print(f"Imagen procesada: {nuevo_nombre}")
                
            except Exception as e:
                print(f"Error procesando {img_nombre}: {str(e)}")
            finally:
                # Limpiar temporales
                for path in [tmp_path, tmp_modified_path]:
                    if path and os.path.exists(path):
                        os.remove(path)

    except Exception as e:
        print(f"Error general: {str(e)}")

# ======================
# Ejecución
# ======================
if __name__ == "__main__":
    try:
        supabase.storage.get_bucket(SOURCE_BUCKET)
        supabase.storage.get_bucket(DESTINATION_BUCKET)
        procesar_y_mover_imagen()
        print("Proceso completado exitosamente")
    except Exception as e:
        print(f"Error inicial: {str(e)}")
        print("Verifica que los buckets existen y las políticas de acceso")
