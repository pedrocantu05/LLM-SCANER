#PASO 6: Integración de todos los pasos en un pipeline completo

import json
import os
import ollama
from preprocesamiento import capturar_y_procesar
from guardar import guardar_resultados

#RUTAS DE ARCHIVOS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_VLM = "qwen2.5vl:7b"
RUTA_ESQUEMA = os.path.join(BASE_DIR, "esquemas", "maestro.json")
RUTA_MODELO = os.path.join(BASE_DIR, "best.pt")

def extraer_datos_automatico(ruta_imagen):
    with open(RUTA_ESQUEMA, "r", encoding="utf-8") as f:
        esquema_json = json.load(f)

    print(f"\n Clasificando y extrayendo datos con {MODELO_VLM} ---")

    prompt = (
        "Eres un sistema OCR de alta precisión para identificaciones oficiales. "
        "1. Clasifica el documento obligatoriamente como 'INE' o 'VISA_AMERICANA'. "
        "   - Si ves textos en inglés como 'USA', 'B1/B2', 'SURNAME', 'GIVEN NAMES', 'DATE OF ISSUE' o 'EXPIRES ON', es 'VISA_AMERICANA', NUNCA 'INE'. "
        "   - Si ves 'INSTITUTO NACIONAL ELECTORAL', 'CREDENCIAL PARA VOTAR', 'CURP' o 'CLAVE DE ELECTOR', es 'INE'. "
        "2. Extrae los datos respetando exactamente letras, números y fechas. "
        "3. Si un campo pertenece a este documento pero no es legible, coloca 'NO_VISIBLE'. "
        "4. Si un campo es exclusivo del otro tipo de documento (por ejemplo, 'tipo_visa' en una INE "
        "o 'clave_elector' en una Visa), coloca 'NO_APLICA'."
    )

    respuesta = ollama.chat(
        model=MODELO_VLM,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [ruta_imagen]
            }
        ],
        format=esquema_json,
        options={"temperature": 0.0}
    )

    datos_completos = json.loads(respuesta["message"]["content"])
    datos_limpios = {
        k: v for k, v in datos_completos.items()
        if v not in ("NO_APLICA", "N/A", "")
    }
    return datos_completos, datos_limpios

def ejecutar_pipeline():
    print("LLM ID SCANNER")
    total_guardados = 0

    while True:
        # PREPROCESAMIENTO: Captura con iPhone + YOLO keypoints + Perspectiva + CLAHE
        ruta_imagen_limpia = capturar_y_procesar(RUTA_MODELO)

        if not ruta_imagen_limpia or not os.path.exists(ruta_imagen_limpia):
            print("\nCaptura cancelada o ventana cerrada.")
            break

        # OLLAMA: Clasificación automática + Extracción con Ollama (Qwen-VL)
        datos_completos, datos_limpios = extraer_datos_automatico(ruta_imagen_limpia)
        print(f"\n DOCUMENTO DETECTADO: {datos_limpios.get('tipo_documento')} <<<")
        print(json.dumps(datos_limpios, indent=4, ensure_ascii=False))

        # GUARDAR: Guarda el JSON individual y agrega la fila al CSV maestro
        guardar_resultados(datos_completos)
        total_guardados += 1
        opcion = input("\n¿Quieres escanear otro documento? [ENTER = Sí | q = Terminar]: ").strip().lower()
        if opcion == "q":
            break

    print(f"\nSesión finalizada. Total de documentos registrados en el CSV: {total_guardados}")

if __name__ == "__main__":
    ejecutar_pipeline()