LLM-SCANER

Escáner inteligente de identificaciones que utiliza YOLO para la detección de documentos y reconocimiento de esquinas en tiempo real, OpenCV para la corrección de perspectiva y mejora de imágenes, y Ollama (Qwen 2.5-VL) para el reconocimiento óptico de caracteres (OCR), clasificación de documentos y extracción estructurada de información. Exporta automáticamente los datos obtenidos a archivos JSON y CSV.

Visión Computacional | 608984 | Pedro Daniel Cantú Lozano

⸻

Descripción del proyecto

LLM-SCANER es un sistema automatizado para escanear y extraer información de documentos de identificación, específicamente credenciales INE y visas americanas.

El proyecto utiliza YOLO para detectar las cuatro esquinas del documento mediante una cámara web. Posteriormente, OpenCV corrige la perspectiva y mejora la calidad de la imagen para facilitar el reconocimiento de texto.

Finalmente, Ollama, mediante el modelo Qwen 2.5-VL, analiza la imagen, identifica el tipo de documento y extrae la información relevante en formato JSON. Los datos obtenidos también se almacenan en un archivo CSV para su consulta y procesamiento posterior.

⸻

Desarrollo del proyecto

PRE-PROYECTO (Proceso de entrenamiento)

Esta etapa corresponde a la preparación del dataset y entrenamiento del modelo YOLO. No forma parte de los archivos principales del repositorio de GitHub.

PASO 1: Preparación de datos

* Archivo: descargar.py
* Dataset: Roboflow - ID Card Dataset
* Objetivo: Descargar y preparar el conjunto de imágenes que se utilizará durante el entrenamiento del modelo.
* Resultado: Dataset descargado y organizado para el entrenamiento de YOLO.

PASO 2: Verificación de etiquetas (Labels)

* Archivo: verificar_labels.py
* Objetivo: Verificar que las etiquetas correspondan correctamente a las cuatro esquinas de las credenciales, asegurando su ubicación y orientación.
* Resultado: Identificación y eliminación de imágenes con etiquetas incorrectas para evitar errores durante el entrenamiento.

PASO 3: Entrenamiento del modelo YOLO

* Archivo: train.py
* Objetivo: Entrenar un modelo YOLO capaz de detectar las cuatro esquinas de una credencial.
* Configuración: Se utiliza kpt_shape = [4, 3], donde cada punto contiene sus coordenadas (x, y) y su indicador de visibilidad.
* Resultado: Generación del modelo entrenado best.pt, que posteriormente se utiliza para detectar las esquinas de los documentos mediante la cámara web.

⸻

PROYECTO (Repositorio GitHub)

PASO 4: Captura, detección de esquinas y procesamiento de imagen

* Archivo: preprocesamiento.py
* Modelo: best.pt

Objetivo: Capturar la identificación mediante una cámara web y obtener una imagen estandarizada que facilite la extracción de información.

Proceso:

1. Captura de video en tiempo real mediante OpenCV.
2. Detección de las cuatro esquinas del documento utilizando YOLO.
3. Estabilización de las coordenadas mediante suavizado EMA (Exponential Moving Average).
4. Evaluación de calidad de imagen considerando nitidez, iluminación, reflejos, movimiento y tamaño del documento.
5. Corrección de perspectiva mediante cv2.getPerspectiveTransform() y cv2.warpPerspective(), estandarizando las dimensiones de la credencial a 856 × 540 píxeles.
6. Mejora de imagen mediante técnicas de Gamma Correction, CLAHE, Denoising, filtro bilateral y Unsharp Mask.

Resultado: Generación de tres imágenes:

Archivo	Descripción
1_original.jpg	Captura original de la cámara web.
2_perspectiva.jpg	Documento con perspectiva corregida y dimensiones estandarizadas.
3_limpia_clahe.jpg	Imagen final procesada y optimizada para el reconocimiento de texto mediante Ollama.

PASO 5: Reconocimiento de texto y extracción de información

* Archivo: main.py
* Modelo: Ollama - Qwen 2.5-VL

Objetivo: Analizar la imagen procesada para identificar el tipo de documento y extraer automáticamente sus datos.

Proceso:

1. Recepción de la imagen final generada por preprocesamiento.py.
2. Envío de la imagen al modelo Qwen 2.5-VL mediante Ollama.
3. Clasificación del documento como INE o VISA AMERICANA.
4. Reconocimiento y extracción de los campos de información correspondientes.
5. Organización de los datos en una estructura JSON siguiendo un esquema predefinido.

Resultado: Generación de archivos JSON individuales con la información extraída de cada identificación.

PASO 6: Almacenamiento y exportación de resultados

* Archivo: guardar.py

Objetivo: Almacenar de manera estructurada la información obtenida durante el proceso de reconocimiento.

Proceso:

1. Recepción de los datos extraídos por Ollama.
2. Generación de un archivo JSON individual para cada identificación.
3. Registro de la información en un archivo CSV maestro.
4. Actualización del historial de identificaciones procesadas.

Resultado:

* registros_identificaciones.csv: Archivo maestro que concentra los registros de todas las identificaciones procesadas.
* INE_JSON: Archivos JSON individuales correspondientes a credenciales INE.
* VISA_JSON: Archivos JSON individuales correspondientes a visas americanas.

⸻

Tecnologías utilizadas

* Python: Lenguaje principal de programación.
* YOLO: Detección de documentos y reconocimiento de esquinas.
* OpenCV: Captura, corrección de perspectiva y procesamiento de imágenes.
* Ollama: Ejecución local del modelo de inteligencia artificial.
* Qwen 2.5-VL: Reconocimiento de texto, clasificación de documentos y extracción de información.
* JSON: Estructuración y almacenamiento individual de los datos.
* CSV: Almacenamiento consolidado de registros.
