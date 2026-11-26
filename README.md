# LLM-SCANER

Escáner inteligente de identificaciones que utiliza **YOLO** para la detección de documentos y reconocimiento de esquinas en tiempo real, **OpenCV** para la corrección de perspectiva y mejora de imágenes, y **Ollama (Qwen 2.5-VL)** para el reconocimiento óptico de caracteres (OCR), clasificación de documentos y extracción estructurada de información. Exporta automáticamente los datos obtenidos a archivos **JSON y CSV**, todo **de forma local, sin internet**.

**Visión Computacional | 608984 | Pedro Daniel Cantú Lozano | pedro.cantul@udem.edu**

---

## Escenario

En el mostrador de documentación de un aeropuerto **se cae el sistema**. Las filas crecen, los vuelos no esperan y capturar a mano la INE o la visa de cada pasajero es lento y propenso a errores; además, sin red no se pueden usar servicios en la nube.

**LLM-SCANER** permite seguir registrando pasajeros **de forma local**: cada identificación se escanea con una cámara, se lee con un modelo que corre en la misma computadora y se guarda en JSON y CSV. Cuando el sistema regresa, el archivo `registros_identificaciones.csv` se importa a la base de datos. Los datos **nunca salen de la computadora**.

```
Cámara → YOLOv8-pose (best.pt) → OpenCV (esquinas, homografía, mejora) → Qwen 2.5-VL en Ollama → JSON + CSV
```

---

## Descripción del proyecto

LLM-SCANER es un sistema automatizado para escanear y extraer información de documentos de identificación, específicamente **credenciales INE** y **visas americanas**.

El proyecto utiliza YOLO para detectar las cuatro esquinas del documento mediante una cámara. Posteriormente, OpenCV refina esas esquinas, corrige la perspectiva y mejora la calidad de la imagen para facilitar el reconocimiento de texto.

Finalmente, Ollama, mediante el modelo Qwen 2.5-VL, analiza la imagen, identifica el tipo de documento y extrae la información relevante en formato JSON. Los datos obtenidos también se almacenan en un archivo CSV para su consulta y procesamiento posterior.

---

## Estructura del repositorio

```
├── main.py                 # Ejecuta todo el flujo
├── preprocesamiento.py     # Cámara, detección, esquinas, homografía y mejora de imagen
├── guardar.py              # Guarda el JSON individual y la fila del CSV
├── best.pt                 # Modelo YOLOv8n-pose entrenado (credencial + 4 esquinas)
├── esquemas/
│   └── maestro.json        # Esquema JSON que Qwen debe llenar (campos de INE y visa)
├── requirements.txt
└── README.md
```

> `maestro.json` debe estar dentro de la carpeta `esquemas/`, porque `main.py` lo busca en esa ruta.

---

## Requisitos

- **Python 3.10** o superior
- **[Ollama](https://ollama.com)** instalado y abierto
- Una **cámara**: webcam, cámara USB o iPhone con Continuity Camera
- Probado en **Mac con Apple Silicon** (`DEVICE = "mps"` en `preprocesamiento.py`). En Windows o Linux, cambia `DEVICE` a `"cpu"` (o a `0` si tienes GPU NVIDIA).

## Instalación

```bash
git clone <url-del-repo>
cd <carpeta-del-repo>

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ollama pull qwen2.5vl:7b
```

## Uso

```bash
python3 main.py
```

1. **Revisión inicial:** el programa verifica que existan `best.pt` y `esquemas/maestro.json`, y que Ollama tenga el modelo; si falta algo, avisa antes de abrir la cámara.
2. **Elige la cámara:** `0` webcam, `1` genérica, `2` iPhone. Cada una carga su propio perfil de umbrales.
3. **Muestra la credencial:** cuando está cerca, quieta, sin reflejo y nítida durante **1 segundo**, se captura sola. También puedes presionar `s` para capturar a mano; `q` o `Esc` cancela.
4. **Qwen lee el documento:** lo clasifica como INE o visa y extrae sus datos. Si falla, puedes reintentar con la misma imagen.
5. **¿Otro documento?** `ENTER` escanea otro; `q` termina y muestra el resumen de la sesión.

---

## Desarrollo del proyecto

### PRE-PROYECTO (Proceso de entrenamiento)

Esta etapa corresponde a la preparación del dataset y entrenamiento del modelo YOLO. **No forma parte de los archivos del repositorio**; su resultado es el modelo `best.pt`.

**PASO 1: Preparación de datos**
- Archivo: `descargar.py`
- Dataset: Roboflow – ID Card Dataset (300 imágenes)
- Objetivo: Descargar y preparar el conjunto de imágenes que se utilizará durante el entrenamiento del modelo.

**PASO 2: Verificación de etiquetas (labels)**
- Archivo: `verificar_labels.py`
- Objetivo: Verificar que las etiquetas de las cuatro esquinas vengan en el orden correcto (TL, TR, BR, BL).
- Resultado: Se conservaron 213 de 300 imágenes (151 train / 41 valid / 21 test); las demás se eliminaron para evitar errores durante el entrenamiento.

**PASO 3: Entrenamiento del modelo YOLO**
- Archivo: `train.py`
- Objetivo: Entrenar YOLOv8n-pose para detectar la credencial y sus cuatro esquinas.
- Configuración: `kpt_shape = [4, 3]` (x, y y visibilidad de cada esquina), aumentos de rotación (`degrees = 15`) y perspectiva (`perspective = 0.0007`).
- Resultado: `best.pt`, con mAP50-95 de pose de **0.957** en validación.

### PROYECTO (Repositorio GitHub)

**PASO 4: Captura, detección de esquinas y procesamiento de imagen**
- Archivo: `preprocesamiento.py`
- Modelo: `best.pt`
- Objetivo: Capturar la identificación con la cámara y obtener una imagen estandarizada que facilite la extracción de información.

Proceso:
1. Captura de video en tiempo real mediante OpenCV.
2. Detección de las cuatro esquinas del documento utilizando YOLO.
3. Estabilización de las coordenadas mediante suavizado EMA adaptativo (Exponential Moving Average).
4. Refinamiento de las esquinas con OpenCV (GrabCut y líneas de Hough) para que la homografía sea exacta.
5. Evaluación de calidad de imagen: nitidez, iluminación, reflejos, movimiento y tamaño del documento.
6. Corrección de perspectiva mediante `cv2.getPerspectiveTransform()` y `cv2.warpPerspective()`, estandarizando la credencial a **856 × 540 píxeles** (solo si no está recta).
7. Mejora de imagen mediante Gamma Correction, CLAHE, Denoising, filtro bilateral y Unsharp Mask.

Resultado: cada escaneo se guarda en su propia carpeta `capturas/imagen_N/`:

| Archivo | Descripción |
|---|---|
| `0_deteccion.jpg` | Captura con las esquinas detectadas y un panel con los parámetros de calidad. |
| `1_original.jpg` | Captura original de la cámara. |
| `2_homografia.jpg` | Documento con perspectiva corregida y dimensiones estandarizadas. |
| `3_limpia.jpg` | Imagen final procesada y optimizada para el reconocimiento de texto mediante Ollama. |
| `datos.json` | Datos extraídos de ese documento. |

**PASO 5: Reconocimiento de texto y extracción de información**
- Archivo: `main.py`
- Modelo: Ollama – Qwen 2.5-VL
- Objetivo: Analizar la imagen procesada para identificar el tipo de documento y extraer automáticamente sus datos.

Proceso:
1. Recepción de la imagen final generada por `preprocesamiento.py`.
2. Envío de la imagen al modelo Qwen 2.5-VL mediante Ollama, con temperatura 0.
3. Clasificación del documento como INE o VISA AMERICANA.
4. Reconocimiento y extracción de los campos correspondientes (`NO_VISIBLE` si no se lee, `NO_APLICA` si es del otro documento).
5. Organización de los datos en una estructura JSON siguiendo el esquema `esquemas/maestro.json`.

**PASO 6: Almacenamiento y exportación de resultados**
- Archivo: `guardar.py`
- Objetivo: Almacenar de manera estructurada la información obtenida durante el proceso de reconocimiento.

Proceso:
1. Recepción de los datos extraídos por Ollama.
2. Normalización del registro (los campos del otro tipo de documento quedan como `NO_APLICA`).
3. Generación de un archivo JSON individual para cada identificación.
4. Registro de la información en el CSV maestro.

Resultado (carpeta `resultados/`):
- `registros_identificaciones.csv`: archivo maestro con los registros de todas las identificaciones procesadas, listo para importar a la base de datos.
- `INE_<fecha>_<hora>.json`: archivos JSON individuales de credenciales INE.
- `VISA_AMERICANA_<fecha>_<hora>.json`: archivos JSON individuales de visas americanas.

> Las carpetas `capturas/` y `resultados/` contienen **datos personales**: no las subas al repositorio.

---

## Tecnologías utilizadas

- **Python**: lenguaje principal de programación.
- **YOLO (YOLOv8n-pose)**: detección de documentos y reconocimiento de esquinas.
- **OpenCV**: captura, refinamiento de esquinas, corrección de perspectiva y procesamiento de imágenes.
- **Ollama**: ejecución local del modelo de inteligencia artificial.
- **Qwen 2.5-VL**: reconocimiento de texto, clasificación de documentos y extracción de información.
- **JSON**: estructuración y almacenamiento individual de los datos.
- **CSV**: almacenamiento consolidado de registros.
