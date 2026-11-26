#PASO 5: Guardado de resultados en CSV y JSON

import csv
import json
import os
from datetime import datetime

COLUMNAS_CSV = [
    "fecha_escaneo",
    "tipo_documento",
    "nombre_completo",
    "fecha_nacimiento",
    "sexo",
    "vigencia",
    "domicilio",
    "clave_elector",
    "curp",
    "año_de_registro",
    "seccion",
    "nacionalidad",
    "fecha_expedicion",
    "tipo_visa"
]

CAMPOS_SOLO_VISA = ["nacionalidad", "fecha_expedicion", "tipo_visa"]
CAMPOS_SOLO_INE = ["domicilio", "clave_elector", "curp", "año_de_registro", "seccion"]

def normalizar_registro(datos_raw):
    #Asegura que los campos exclusivos del otro documento queden como NO_APLICA.
    datos = datos_raw.copy()
    tipo = datos.get("tipo_documento", "")

    #Si el tipo de documento no es reconocido, se asigna "NO_APLICA" a todos los campos específicos.
    if tipo not in ["INE", "VISA_AMERICANA"]:
        for campo in CAMPOS_SOLO_VISA + CAMPOS_SOLO_INE:
            datos[campo] = "NO_APLICA"
    elif tipo == "INE":
        for campo in CAMPOS_SOLO_VISA:
            datos[campo] = "NO_APLICA"
    elif tipo == "VISA_AMERICANA":
        for campo in CAMPOS_SOLO_INE:
            datos[campo] = "NO_APLICA"

    #Fecha en la que se escanea el documento, para registro histórico
    registro_ordenado = {
        "fecha_escaneo": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    for col in COLUMNAS_CSV[1:]:
        valor = datos.get(col, "NO_APLICA")
        # Si el LLM devolvió "N/A" o vacío, lo unifica como "NO_APLICA"
        if not valor or str(valor).strip().upper() == "N/A":
            valor = "NO_APLICA"
        registro_ordenado[col] = valor

    return registro_ordenado

def guardar_resultados(datos_raw, carpeta_salida="resultados"):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta_carpeta = os.path.join(BASE_DIR, carpeta_salida)
    os.makedirs(ruta_carpeta, exist_ok=True)

    registro_csv = normalizar_registro(datos_raw)
    tipo_doc = registro_csv["tipo_documento"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Guardar archivo JSON individual (únicamente con los campos del documento detectado)
    datos_json_limpio = {k: v for k, v in registro_csv.items() if v != "NO_APLICA"}
    ruta_json = os.path.join(ruta_carpeta, f"{tipo_doc}_{timestamp}.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(datos_json_limpio, f, indent=4, ensure_ascii=False)

    # 2. Agregar fila al CSV maestro (utf-8-sig evita símbolos raros con acentos/Ñ en Excel)
    ruta_csv = os.path.join(ruta_carpeta, "registros_identificaciones.csv")
    existe_csv = os.path.isfile(ruta_csv)
    with open(ruta_csv, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        if not existe_csv:
            writer.writeheader()
        writer.writerow(registro_csv)

    print("\nResultados guardados exitosamente:")
    print(f"JSON individual guardado en: {ruta_json}")
    print(f"Fila agregada en CSV maestro: {ruta_csv}")
    return ruta_json, ruta_csv