# Prototipo de Microcontrolador para Detección de Desechos Sólidos

## Descripción

Este proyecto tiene como objetivo desarrollar un prototipo de microcontrolador para la detección de desechos sólidos en la orilla del Lago Coquivacoa, específicamente en el Colegio Santa Rosa. El sistema utiliza un ESP32cam programado en C para tomar imágenes y enviarlas a Supabase, donde una aplicación en Python procesa las imágenes con un modelo de inteligencia artificial (YOLOv5) y las almacena en un bucket diferente.

## Instalación

1. **Hardware:**
   - ESP32cam
   - Cámara compatible

2. **Software:**
   - Python 3.9
   - Supabase account

3. **Configuración del ESP32cam:**
   - Programar el ESP32cam para capturar imágenes y enviarlas a Supabase.

4. **Configuración de la aplicación:**
   - Configurar la aplicación para buscar imágenes en Supabase, procesarlas con el modelo YOLOv5 y almacenarlas en el bucket correspondiente.

## Uso

1. Iniciar el ESP32cam para comenzar a capturar imágenes.
2. Ejecutar la aplicación para procesar las imágenes y almacenarlas en el bucket de Supabase.
3. Monitorear los resultados a través del sistema de visualización en tiempo real.


## Autores

- **Desarrolladores:**
  - Rodolfo Rodríguez
  - José Villalobos

- **Investigadores:**
  - Arlany Machado
  - Gabriel Rodríguez
  - Karibell Bracho
  - Oliver Morán

## Agradecimientos

Agradecemos al equipo de **Samsung Innovation Campus** por brindarnos los conocimientos necesarios para desarrollar este proyecto. También extendemos nuestro reconocimiento a todas las personas y organizaciones que han contribuido de alguna manera a este proyecto. En particular, reconocemos la valiosa ayuda de los investigadores que nos proporcionaron datos y conocimientos sobre la contaminación del Lago de Maracaibo.
