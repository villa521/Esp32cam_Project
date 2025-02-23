import os
import numpy as np
import tensorflow as tf
from supabase import create_client
from tensorflow.keras.preprocessing import image
import tempfile
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# ======================
# Configuración
# ======================
SUPABASE_URL = "https://xshchsisefefyazmgewl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhzaGNoc2lzZWZlZnlhem1nZXdsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg0NTAwNjYsImV4cCI6MjA1NDAyNjA2Nn0.OGxuHQ_ApZpC27APq2GdpXuMtACyTqOkwr-DXzC4lT4"
SOURCE_BUCKET = "objetos"
DESTINATION_BUCKET = "deteccion"
IMG_SIZE = (300, 300)
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
MODEL_PATH = r'C:\Users\Villa5050G\Desktop\model.tflite'

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
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

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

def procesar_imagen_tflite(ruta_imagen):
    """Realiza la predicción con el modelo TFLite"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = image.load_img(ruta_imagen, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    return CLASSES[np.argmax(pred)]

def get_timestamp():
    """Genera un timestamp para evitar colisiones de nombres"""
    return datetime.now().strftime("%Y%m%d%H%M%S")

def anadir_texto_a_imagen(ruta_imagen, texto):
    """Añade texto legible con fondo a la imagen"""
    try:
        img = Image.open(ruta_imagen)
        draw = ImageDraw.Draw(img)
        
        # Cargar fuente con fallbacks
        try:
            font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", FONT_SIZE)
            except:
                font = ImageFont.load_default(FONT_SIZE)
        
        # Calcular tamaño del texto
        text_bbox = draw.textbbox(TEXT_POSITION, f"Clasificación: {texto}", font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Dibujar fondo
        draw.rectangle(
            [
                TEXT_POSITION[0] - BORDER,
                TEXT_POSITION[1] - BORDER,
                TEXT_POSITION[0] + text_width + BORDER,
                TEXT_POSITION[1] + text_height + BORDER
            ],
            fill=BACKGROUND_COLOR
        )
        
        # Dibujar texto
        draw.text(
            TEXT_POSITION,
            f"Clasificación: {texto}",
            font=font,
            fill=TEXT_COLOR,
            stroke_width=2,
            stroke_fill=(255, 0, 0)
        )
        
        return img
    except Exception as e:
        print(f"Error añadiendo texto: {str(e)}")
        return None

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
    """Procesa y mueve las imágenes con etiqueta grande"""
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
                
                prediccion = procesar_imagen_tflite(tmp_path)
                print(f"Predicción: {img_nombre} -> {prediccion}")
                
                # Añadir texto
                img_con_etiqueta = anadir_texto_a_imagen(tmp_path, prediccion)
                
                if img_con_etiqueta:
                    # Generar nombres
                    nombre_base, extension = os.path.splitext(img_nombre)
                    nuevo_nombre = f"{prediccion}_{nombre_base}_{get_timestamp()}{extension}"
                    
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