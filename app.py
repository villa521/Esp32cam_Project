import tkinter as tk
from tkinter import messagebox
import os
import tempfile
from datetime import datetime
from PIL import Image
import cv2  # Asegúrate de tener instalado opencv-python
from supabase import create_client
from ultralytics import YOLO  # Importar YOLO de ultralytics

# ======================
# Configuración
# ======================
# Datos de conexión a Supabase
SUPABASE_URL = "https://xshchsisefefyazmgewl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhzaGNoc2lzZWZlZnlhem1nZXdsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg0NTAwNjYsImV4cCI6MjA1NDAyNjA2Nn0.OGxuHQ_ApZpC27APq2GdpXuMtACyTqOkwr-DXzC4lT4"
SOURCE_BUCKET = "objetos"
DESTINATION_BUCKET = "deteccion"

# Usar ruta relativa para el modelo (se debe ubicar en la carpeta principal del proyecto)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best2.pt")

# ======================
# Inicializar clientes
# ======================
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
model = YOLO(MODEL_PATH)  # Cargar modelo YOLOv5

# ======================
# Funciones auxiliares
# ======================
def descargar_imagen_temporal(nombre_archivo):
    """Descarga una imagen a un archivo temporal."""
    try:
        file_data = supabase.storage.from_(SOURCE_BUCKET).download(nombre_archivo)
        _, ext = os.path.splitext(nombre_archivo)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_data)
            return tmp_file.name
    except Exception as e:
        print(f"Error descargando {nombre_archivo}: {str(e)}")
        return None

def procesar_imagen_yolov5(ruta_imagen):
    """Realiza la detección con el modelo YOLOv5."""
    results = model(ruta_imagen)
    results_img = results[0].plot()  # Obtener la imagen con detecciones
    # Convertir imagen BGR a RGB para PIL
    image_rgb = cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

def get_timestamp():
    """Genera un timestamp para evitar colisiones en los nombres."""
    return datetime.now().strftime("%Y%m%d%H%M%S")

def guardar_imagen_temporal(imagen_pil, extension):
    """Guarda la imagen modificada en un archivo temporal."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            formato = "JPEG" if extension.lower() in ['.jpg', '.jpeg'] else "PNG"
            imagen_pil.save(tmp_file, format=formato, quality=95)
            return tmp_file.name
    except Exception as e:
        print(f"Error guardando imagen temporal: {str(e)}")
        return None

def subir_imagen(ruta_local, nombre_destino):
    """Sube la imagen modificada al bucket destino."""
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
# Función principal de procesamiento
# ======================
def procesar_y_mover_imagen(text_widget):
    """Procesa y mueve las imágenes con detección YOLOv5."""
    try:
        archivos = supabase.storage.from_(SOURCE_BUCKET).list()
        imagenes = [f['name'] for f in archivos if f['name'].lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not imagenes:
            text_widget.insert(tk.END, "No se encontraron imágenes para procesar.\n")
            return

        text_widget.insert(tk.END, f"Procesando {len(imagenes)} imágenes...\n")
        
        for img_nombre in imagenes:
            tmp_path = None
            tmp_modified_path = None
            try:
                # Descargar y procesar la imagen
                tmp_path = descargar_imagen_temporal(img_nombre)
                if not tmp_path:
                    continue
                
                img_con_etiqueta = procesar_imagen_yolov5(tmp_path)
                
                if img_con_etiqueta:
                    # Generar nombre para la imagen procesada
                    nombre_base, extension = os.path.splitext(img_nombre)
                    nuevo_nombre = f"deteccion_{nombre_base}_{get_timestamp()}{extension}"
                    
                    # Guardar imagen procesada en un archivo temporal
                    tmp_modified_path = guardar_imagen_temporal(img_con_etiqueta, extension)
                    
                    # Subir la imagen modificada y eliminar la original
                    if subir_imagen(tmp_modified_path, nuevo_nombre):
                        supabase.storage.from_(SOURCE_BUCKET).remove([img_nombre])
                        text_widget.insert(tk.END, f"Imagen procesada: {nuevo_nombre}\n")
            except Exception as e:
                text_widget.insert(tk.END, f"Error procesando {img_nombre}: {str(e)}\n")
            finally:
                # Eliminar archivos temporales
                for path in [tmp_path, tmp_modified_path]:
                    if path and os.path.exists(path):
                        os.remove(path)
    except Exception as e:
        text_widget.insert(tk.END, f"Error general: {str(e)}\n")

# ======================
# Clase de la aplicación (GUI)
# ======================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("App de Procesamiento de Imágenes")
        self.root.geometry("600x400")

        # Variable para el intervalo de ejecución (en minutos)
        self.sleep_time = tk.IntVar(value=0)
        self.auto_interval = None  # Intervalo en minutos para ejecución automática
        self.after_id = None  # ID del 'after' programado (para poder cancelarlo)

        # Área de texto que simula una terminal para mostrar logs
        self.text_widget = tk.Text(root, height=15, width=70)
        self.text_widget.pack(pady=10)
        
        # Botón para ejecutar el procesamiento manualmente
        self.run_button = tk.Button(root, text="Run (Manual)", command=self.run_process_once)
        self.run_button.pack(pady=5)

        # Configuración del intervalo para ejecución automática
        self.sleep_label = tk.Label(root, text="Intervalo (minutos):")
        self.sleep_label.pack()

        self.sleep_entry = tk.Entry(root, textvariable=self.sleep_time)
        self.sleep_entry.pack(pady=5)

        self.sleep_button = tk.Button(root, text="Set Interval", command=self.set_interval)
        self.sleep_button.pack(pady=5)

        # Botón para detener el bucle automático
        self.stop_button = tk.Button(root, text="Stop Loop", command=self.stop_loop)
        self.stop_button.pack(pady=5)

    def run_process_once(self):
        self.text_widget.insert(tk.END, "Iniciando procesamiento manual...\n")
        self.root.after(100, self.start_processing_once)

    def start_processing_once(self):
        procesar_y_mover_imagen(self.text_widget)
        self.text_widget.insert(tk.END, "Procesamiento manual finalizado.\n")

    def run_process_loop(self):
        self.text_widget.insert(tk.END, "Iniciando procesamiento automático...\n")
        procesar_y_mover_imagen(self.text_widget)
        self.text_widget.insert(tk.END, "Procesamiento automático finalizado.\n")
        # Programar la siguiente ejecución si se mantiene el intervalo
        if self.auto_interval:
            self.after_id = self.root.after(self.auto_interval * 60000, self.run_process_loop)

    def set_interval(self):
        try:
            interval = int(self.sleep_time.get())
            if interval > 0:
                self.auto_interval = interval
                self.text_widget.insert(tk.END, f"Intervalo establecido en {interval} minutos. Iniciando bucle automático...\n")
                # Cancelar un bucle ya programado (si existe)
                if self.after_id:
                    self.root.after_cancel(self.after_id)
                # Programar la primera ejecución automática después del intervalo
                self.after_id = self.root.after(self.auto_interval * 60000, self.run_process_loop)
            else:
                self.text_widget.insert(tk.END, "Por favor, ingrese un valor válido mayor que 0.\n")
        except ValueError:
            self.text_widget.insert(tk.END, "Por favor, ingrese un número entero válido para el intervalo.\n")

    def stop_loop(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
            self.auto_interval = None
            self.text_widget.insert(tk.END, "Bucle automático detenido.\n")
        else:
            self.text_widget.insert(tk.END, "No hay un bucle automático en ejecución.\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
