#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// Configuración de la cámara (para ESP32-CAM AI-Thinker)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define FLASH_GPIO_NUM     4  // Pin del flash

// Credenciales Wi-Fi
const char* ssid = "SALA";
const char* password = "12345678";

// Configuración Supabase
const String SUPABASE_URL = "https://xshchsisefefyazmgewl.supabase.co";
const String SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhzaGNoc2lzZWZlZnlhem1nZXdsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg0NTAwNjYsImV4cCI6MjA1NDAyNjA2Nn0.OGxuHQ_ApZpC27APq2GdpXuMtACyTqOkwr-DXzC4lT4";
const char* bucketName = "objetos";

// Variables para el control del tiempo
unsigned long nextPhotoTime = 0;
unsigned long lastCountdownUpdate = 0;

void subirImagenSupabase(uint8_t* imagen, size_t tamanio, String nombreArchivo);
void guardarEnBaseDeDatos(String imageUrl);
String generarUUID();
String getCurrentTimestamp();

void setup() {
  Serial.begin(115200);
  
  // Inicializar flash
  pinMode(FLASH_GPIO_NUM, OUTPUT);
  digitalWrite(FLASH_GPIO_NUM, LOW);

  // Conexión WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi conectado");

  // Configuración de la cámara
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_UXGA;
  config.jpeg_quality = 10;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Error al iniciar la cámara: 0x%x", err);
    while (true);
  }
}

void loop() {
  // Verificar si es hora de tomar una foto
  if (millis() >= nextPhotoTime) {
    // Activar flash
    digitalWrite(FLASH_GPIO_NUM, HIGH);
    delay(50); // Breve espera para estabilización
    
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Error al capturar la imagen");
      digitalWrite(FLASH_GPIO_NUM, LOW);
      return;
    }
    digitalWrite(FLASH_GPIO_NUM, LOW); // Apagar flash

    String nombreArchivo = "foto_" + String(millis()) + ".jpg";
    subirImagenSupabase(fb->buf, fb->len, nombreArchivo);
    esp_camera_fb_return(fb);

    nextPhotoTime = millis() + 60000; // Programar próxima foto
  }

  // Mostrar contador cada segundo
  if (millis() - lastCountdownUpdate >= 1000) {
    lastCountdownUpdate = millis();
    long remaining = (nextPhotoTime - millis()) / 1000;
    if (remaining < 0) remaining = 0;
    Serial.print("Tiempo restante: ");
    Serial.print(remaining);
    Serial.println(" segundos");
  }
}

// Función para subir la imagen a Supabase Storage
void subirImagenSupabase(uint8_t* imagen, size_t tamanio, String nombreArchivo) {
  HTTPClient http;
  
  // URL completa para subir la imagen
  String urlCompleta = SUPABASE_URL + "/storage/v1/object/" + String(bucketName) + "/" + nombreArchivo;

  http.begin(urlCompleta);
  http.addHeader("Authorization", "Bearer " + SUPABASE_KEY);
  http.addHeader("Content-Type", "image/jpeg");
  
  Serial.println("Subiendo imagen: " + nombreArchivo);
  int httpCode = http.POST(imagen, tamanio);
  
  if (httpCode == HTTP_CODE_OK) {
    Serial.println("Imagen subida exitosamente");

    // Obtener la URL pública de la imagen
    String imageUrl = SUPABASE_URL + "/storage/v1/object/public/" + String(bucketName) + "/" + nombreArchivo;

    // Guardar los datos en la base de datos
    guardarEnBaseDeDatos(imageUrl);
  } else {
    Serial.println("Error al subir la imagen: " + String(httpCode));
    String respuesta = http.getString();
    Serial.println("Respuesta del servidor: " + respuesta);
  }
  http.end();
}

// Función para guardar los datos en la base de datos de Supabase
void guardarEnBaseDeDatos(String imageUrl) {
  HTTPClient http;
  
  // URL de la API de Supabase para la tabla "detections"
  String url = SUPABASE_URL + "/rest/v1/detections";

  http.begin(url);
  http.addHeader("Authorization", "Bearer " + SUPABASE_KEY);
  http.addHeader("apikey", SUPABASE_KEY); // Agregar el encabezado apikey
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Prefer", "return=minimal");

  // Generar un UUID para el campo "id"
  String uuid = generarUUID();

  // Crear el JSON con los datos
  String jsonData = "{\"id\": \"" + uuid + "\", \"image_url\": \"" + imageUrl + "\", \"timestamp\": \"" + getCurrentTimestamp() + "\"}";
  Serial.println("JSON enviado: " + jsonData);

  int httpResponseCode = http.POST(jsonData);
  if (httpResponseCode == HTTP_CODE_CREATED) {
    Serial.println("Datos guardados en la base de datos. Código de respuesta: " + String(httpResponseCode));
  } else {
    Serial.println("Error al guardar los datos. Código de respuesta: " + String(httpResponseCode));
    String respuesta = http.getString();
    Serial.println("Respuesta del servidor: " + respuesta);
  }
  http.end();
}

// Función para generar un UUID simple
String generarUUID() {
  String uuid = "";
  for (int i = 0; i < 16; i++) {
    byte randomValue = random(256);
    char buffer[3];
    sprintf(buffer, "%02x", randomValue);
    uuid += buffer;
    if (i == 3 || i == 5 || i == 7 || i == 9) {
      uuid += "-";
    }
  }
  return uuid;
}

// Función para obtener la fecha y hora actual (formato ISO 8601)
String getCurrentTimestamp() {
  // Obtener la fecha y hora actual (puedes usar un servidor NTP para mayor precisión)
  // Aquí se usa un timestamp simple como ejemplo
  return "2023-10-15T12:00:00"; // Reemplaza con la fecha y hora real
}