# LLM-SCANER

Escáner local de **INE** y **visa americana**: detecta la credencial con la cámara, la endereza y extrae sus datos a **JSON y CSV**, todo sin internet.

**Escenario:** en un aeropuerto se cae el sistema. LLM-SCANER permite seguir registrando pasajeros de forma local y, cuando el sistema regresa, importar el CSV a la base de datos.

## Cómo funciona

```
Cámara → YOLOv8-pose (best.pt) → OpenCV (esquinas, homografía, mejora) → Qwen2.5-VL en Ollama → JSON + CSV
```

| Archivo | Qué hace |
|---|---|
| `main.py` | Ejecuta todo el flujo: requisitos → cámara → captura → Qwen → guardado |
| `preprocesamiento.py` | Cámara en vivo, detección, refinado de esquinas, homografía y mejora de imagen |
| `guardar.py` | Normaliza el registro y escribe el JSON individual y la fila del CSV |
| `esquemas/maestro.json` | Esquema JSON que Qwen debe llenar (campos de INE y visa) |
| `best.pt` | Modelo YOLOv8n-pose entrenado: detecta la credencial y sus 4 esquinas |

## Requisitos

- Python 3.10 o superior
- [Ollama](https://ollama.com) instalado y abierto
- Una cámara (webcam, USB o iPhone con Continuity Camera)
- Probado en Mac con Apple Silicon (`DEVICE = "mps"`). En Windows o Linux, cambia `DEVICE` en `preprocesamiento.py` a `"cpu"` (o `0` si tienes GPU NVIDIA).

## Instalación

```bash
git clone <url-del-repo>
cd <repo>
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen2.5vl:7b
```

## Uso

```bash
python3 main.py
```

1. Elige la cámara: `0` webcam, `1` genérica, `2` iPhone.
2. Muestra la credencial: cuando está cerca, quieta, sin reflejo y nítida durante 1 s, se captura sola (o presiona `s`; `q`/`Esc` cancela).
3. Qwen clasifica el documento (INE o visa) y extrae sus datos.
4. `ENTER` escanea otro documento; `q` termina.

## Salidas

```
capturas/imagen_N/   0_deteccion.jpg · 1_original.jpg · 2_homografia.jpg · 3_limpia.jpg · datos.json
resultados/          <TIPO>_<fecha>.json · registros_identificaciones.csv
```

Estas carpetas contienen **datos personales** y están en `.gitignore`.
