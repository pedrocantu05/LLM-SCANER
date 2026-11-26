# PASO 4: CAPTURA, CONTROL DE CALIDAD Y RECTIFICACION

# NOTA SOBRE EL ORDEN DE LAS ESQUINAS: no se reordenan con reglas
# de posicion. verificar_labels.py solo deja pasar al entrenamiento
# imagenes cuyas 4 esquinas estan en orden [TL, TR, BR, BL]; best.pt
# aprende ese orden y lo devuelve en los keypoints 0, 1, 2 y 3. Las
# esquinas refinadas heredan ese mismo orden (_alinear_con_yolo).

# IDEA CLAVE DE LA HOMOGRAFIA (inspirada en el escaner de LearnOpenCV):
# YOLO solo dice "la tarjeta esta mas o menos aqui"; sus keypoints
# pueden quedar fuera con la tarjeta de lado. 
# Una homografia con esos puntos sale chueca. Por eso
# las esquinas finales NO son las de YOLO: YOLO solo siembra la busqueda
# y las esquinas exactas salen del contorno real de la tarjeta.

import itertools
import os
import subprocess
import sys
import time
from collections import deque
import cv2
import numpy as np

# 1. PARÁMTEROS AJUSTABLES

CAMARA = 0                      # camara por defecto si el usuario solo da ENTER
NOMBRES_CAMARA = {              # Pregunta al usuario "Que camara quieres usar?" y muestra estos nombres
    0: "Webcam de la Mac",
    1: "Camara generica (USB u otra)",
    2: "iPhone (Continuity Camera)",
}
DEVICE = "mps"                  # "GPU de MAC

# CONTROL POR CÁMARA: 
# Cada camara "ve" distinto: la webcam tiene menos resolucion real, mas
# compresion y mas ruido (su nitidez es mucho menor y sus keypoints
# tiemblan mas); el iPhone es nitido y estable. Estos 9 parametros SOLO
# se definen aqui; aplicar_perfil_camara() los activa al abrir la camara.
# Para otra camara, copia un perfil con su numero.
#
#  Parametro            Que controla                       Rango sugerido
#  -------------------  ---------------------------------  --------------
#  CONF_YOLO            confianza para EMPEZAR a seguir     0.35 - 0.65
#  CONF_YOLO_SEGUIR     confianza para NO soltarla          0.20 - 0.40
#  FRAMES_TOLERANCIA    frames sin deteccion sin borrar     6 - 15
#  UMBRAL_NITIDEZ_MIN   nitidez minima para capturar        webcam 15-30 / iPhone 60-100
#  UMBRAL_MOVIMIENTO_MAX px/s maximos para capturar         50 - 100
#  ALPHA_SUAVIZADO      suavizado con la tarjeta quieta     0.10 - 0.30 (menor = mas estable)
#  VEL_QUIETA           px/frame que se tratan como temblor 4 - 8
#  VEL_RAPIDA           px/frame desde los que no se suaviza 20 - 30
#  ALPHA_CORRECCION     suavizado de las esquinas exactas   0.15 - 0.35


PERFILES_CAMARA = {0: {                    # webcam de la Mac
        "CONF_YOLO": 0.43,
        "CONF_YOLO_SEGUIR": 0.30,
        "FRAMES_TOLERANCIA": 12,           # la webcam pierde la tarjeta mas seguido
        "UMBRAL_NITIDEZ_MIN": 10.0,        # enfocada da <20
        "UMBRAL_MOVIMIENTO_MAX": 80.0,     # keypoints mas ruidosos
        "ALPHA_SUAVIZADO": 0.15,
        "VEL_QUIETA": 7.0,
        "VEL_RAPIDA": 28.0,
        "ALPHA_CORRECCION": 0.20,
    },2: {                                   # iPhone (Continuity Camera)
        "CONF_YOLO": 0.45,
        "CONF_YOLO_SEGUIR": 0.35,
        "FRAMES_TOLERANCIA": 8,
        "UMBRAL_NITIDEZ_MIN": 40.0,         # enfocada da <85
        "UMBRAL_MOVIMIENTO_MAX": 60.0,      # keypoints mas estables
        "ALPHA_SUAVIZADO": 0.20,            
        "VEL_QUIETA": 5.0,                  
        "VEL_RAPIDA": 22.0,
        "ALPHA_CORRECCION": 0.25,
    },
}

# Cualquier otra camara (USB, otra webcam, etc.) se trata como calidad
# media-baja: cada parametro es el PROMEDIO entre la webcam y el iPhone.
PERFIL_GENERICO = {
    clave: (round((PERFILES_CAMARA[0][clave] + PERFILES_CAMARA[2][clave]) / 2)
            if isinstance(PERFILES_CAMARA[0][clave], int)
            else round((PERFILES_CAMARA[0][clave] + PERFILES_CAMARA[2][clave]) / 2, 3))
    for clave in PERFILES_CAMARA[0]
}
# Valores por defecto
CONF_YOLO = PERFIL_GENERICO["CONF_YOLO"]
CONF_YOLO_SEGUIR = PERFIL_GENERICO["CONF_YOLO_SEGUIR"]
FRAMES_TOLERANCIA = PERFIL_GENERICO["FRAMES_TOLERANCIA"]
UMBRAL_NITIDEZ_MIN = PERFIL_GENERICO["UMBRAL_NITIDEZ_MIN"]
UMBRAL_MOVIMIENTO_MAX = PERFIL_GENERICO["UMBRAL_MOVIMIENTO_MAX"]
ALPHA_SUAVIZADO = PERFIL_GENERICO["ALPHA_SUAVIZADO"]
VEL_QUIETA = PERFIL_GENERICO["VEL_QUIETA"]
VEL_RAPIDA = PERFIL_GENERICO["VEL_RAPIDA"]
ALPHA_CORRECCION = PERFIL_GENERICO["ALPHA_CORRECCION"]

# P1 Poca iluminacion 
UMBRAL_BAJA_LUZ = 70.0          # brillo medio del cuadro (0-255)
GAMMA = 0.7                     # <1 aclara sombras

# P2 Reflejos y exceso de brillo 
UMBRAL_REFLEJO_MAX = 15.0       # % de pixeles quemados permitido

# P3 Desenfoque 
# (nitidez minima: UMBRAL_NITIDEZ_MIN en PERFILES_CAMARA)
SEGUNDOS_PARA_CAPTURA = 1.0     # segundos con la tarjeta valida; se guarda el frame mas nitido
SEGUNDOS_TOLERANCIA = 0.3       # un frame malo suelto (ruido) no reinicia la cuenta

# P4 Movimiento 
# (UMBRAL_MOVIMIENTO_MAX, ALPHA_SUAVIZADO, VEL_QUIETA y VEL_RAPIDA: en PERFILES_CAMARA)
VENTANA_MOVIMIENTO = 0.30       # segundos de historial para medir el movimiento

# P5 Distancia 
ANCHO_MINIMO_TARJETA = 500      # px del lado mayor (camara 1080p) -> aprox 25cm de distancia 

# P6 Perspectiva, inclinacion y rotacion 
ANCHO_DESTINO = 856             # tamano ID-1 (85.6 x 54 mm)
ALTO_DESTINO = 540
TOL_INCLINACION_GRADOS = 1.0    # bordes vs horizontal/vertical
TOL_ESQUINA_GRADOS = 2.0        # esquinas vs 90 grados
TOL_RAZON_LADOS = 0.03          # diferencia entre lados opuestos (3 %)

# P6 Refinamiento de esquinas (homografia exacta)
REFINAR_ESQUINAS = True
ASPECTO_ID1 = 85.6 / 53.98      # proporcion real de INE / visa (1.586)
ESCALA_VIVO, ITER_VIVO = 0.30, 2       # rapido, para la vista en vivo
DIAG_VIVO = 260                 # en vivo la tarjeta se reduce a ~260 px de diagonal
REFINAR_CADA = 12               # en vivo, GrabCut solo 1 de cada N frames
                                # (ALPHA_CORRECCION: en PERFILES_CAMARA)
ESCALA_FINAL, ITER_FINAL = 0.50, 5     # preciso, solo al guardar
RANGO_AJUSTE_BORDE = 0.008      # cuanto puede moverse cada lado al "pegarse" al borde
TOL_ASPECTO = 0.06              # si la proporcion REAL se aleja mas del 6 %, se corrige
FOCAL_REL = (0.6, 1.5)          # focal plausible de la camara (x ancho de imagen)


# Objeto CLAHE reutilizable: crearlo en cada frame cuesta tiempo y siempre
# usa los mismos parametros (contraste local, mosaico 8x8).
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# 2. UTILIDADES GEOMETRICAS
def _angulo(p, q):
    """Angulo en grados de la recta p->q (0 = horizontal hacia la derecha)."""
    return float(np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0])))

def _dist(p, q):
    return float(np.linalg.norm(np.asarray(q) - np.asarray(p)))

def _angulo_interior(prev, cur, nxt):
    """Angulo interior en 'cur' del triangulo prev-cur-nxt (idealmente 90)."""
    v1 = np.asarray(prev) - np.asarray(cur)
    v2 = np.asarray(nxt) - np.asarray(cur)
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

# 3. P1 - POCA ILUMINACION
def medir_brillo(frame):
    """
    PROBLEMA : decidir si hay poca luz.
    SOLUCION : brillo medio del cuadro en escala de grises (0-255).
    Si es menor que UMBRAL_BAJA_LUZ se activa el gamma antes de
    detectar y, al guardar, el denoise.
    """
    return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))


def crear_tabla_gamma(gamma=GAMMA):
    """
    PROBLEMA : imagen oscura -> el texto pierde contraste y el OCR falla.
    SOLUCION : correccion gamma (< 1 aclara las sombras sin quemar
    las zonas claras), precalculada como tabla LUT de 256 valores para
    aplicarla con cv2.LUT en microsegundos por frame.
    """
    return np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype="uint8")


# 4. P2 - REFLEJOS Y EXCESO DE BRILLO
def medir_reflejos(img):
    """
    PROBLEMA : el exceso de brillo borran el texto; 
    SOLUCION : % de pixeles "quemados": muy brillantes (V > 250) Y casi
    sin color (S < 15), la firma de un reflejo blanco puro.
    Si pasa de UMBRAL_REFLEJO_MAX el frame NO cuenta y se pide inclinar
    la tarjeta (no se corrige digitalmente: inventar pixeles podria
    "crear" caracteres falsos).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    quemado = (hsv[:, :, 2] > 250) & (hsv[:, :, 1] < 15)
    return float(np.sum(quemado) / quemado.size * 100.0)


# 5. P3 - DESENFOQUE
def medir_nitidez(img):
    """
    PROBLEMA : foto desenfocada -> letras borrosas.
    SOLUCION : varianza del Laplaciano. El Laplaciano resalta bordes;
    una imagen nitida tiene muchos bordes fuertes (varianza alta) y una
    borrosa casi ninguno (varianza baja). Se mide sobre la imagen YA
    rectificada para que todas las capturas sean comparables.
    Despues, capturar_y_procesar se queda con el frame MAS nitido de la
    secuencia de frames validos, y mejorar_imagen aplica unsharp mask.
    """
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var()) # Segunda derivada: responde fuerte en bordes nitidos; su varianza mide el enfoque
    #Laplaciano: Sensible a cambios bruscos y es un valor (no binario como Canny)


# 6. P4 - MOVIMIENTO
def suavizar_ema(nuevo, previo, alpha):
    """
    PROBLEMA : los keypoints de YOLO "tiemblan" de un frame a otro y la
    tarjeta se rectifica distinta cada vez.
    SOLUCION : media movil exponencial (EMA):
        suave = alpha * nuevo + (1 - alpha) * previo
    alpha bajo = mas estable pero reacciona mas lento.
    """
    nuevo = np.asarray(nuevo, dtype="float32")
    if previo is None:
        return nuevo
    return alpha * nuevo + (1 - alpha) * previo


def suavizar_adaptativo(nuevo, previo):
    """
    PROBLEMA : el EMA fijo tiene RETRASO. Con la tarjeta en movimiento,
    las esquinas suavizadas se quedan atras de la tarjeta real: la
    homografia recorta mal (se cuela la mano o el fondo y se corta la
    credencial).
    SOLUCION : EMA ADAPTATIVO (idea del filtro "One Euro"). Se mide la
    velocidad de las esquinas (px/frame) y alpha cambia con ella:
      * quieta  (<= VEL_QUIETA)  -> alpha = ALPHA_SUAVIZADO: quita el temblor
      * rapida  (>= VEL_RAPIDA)  -> alpha = 1: sigue a la tarjeta sin retraso
      * en medio                 -> interpolacion lineal
    Asi se obtiene estabilidad cuando importa (al capturar, quieta) y
    cero retraso cuando la tarjeta se mueve.
    """
    nuevo = np.asarray(nuevo, dtype="float32")
    if previo is None:
        return nuevo
    vel = float(np.mean(np.linalg.norm(nuevo - previo, axis=1)))
    f = np.clip((vel - VEL_QUIETA) / (VEL_RAPIDA - VEL_QUIETA), 0.0, 1.0)
    alpha = ALPHA_SUAVIZADO + (1.0 - ALPHA_SUAVIZADO) * f #alpha = valor dado, f = cambio en frames (velocidad)
    return alpha * nuevo + (1 - alpha) * previo


def medir_movimiento_tiempo(historial):
    """
    PROBLEMA : medir el movimiento "por frame" depende de la velocidad de
    la camara (a 60 FPS cada paso es la mitad que a 30 FPS) y el temblor
    natural de los keypoints de YOLO (ruido de 3-8 px) se confunde con
    movimiento: la tarjeta quieta salia como "Mantenla quieta".
    SOLUCION : medir en TIEMPO REAL, no en frames, y de forma robusta:
      * historial = (tiempo, keypoints crudos) de los ultimos
        VENTANA_MOVIMIENTO segundos;
      * se compara la MEDIANA de los primeros frames de la ventana contra
        la de los ultimos (la mediana elimina el temblor de un frame);
      * desplazamiento / tiempo transcurrido = velocidad en px/s.
    El resultado es el mismo a 15, 30 o 60 FPS.
    """
    if len(historial) < 4:
        return 0.0
    tiempos = np.array([t for t, _ in historial])
    puntos = np.array([p for _, p in historial])
    k = max(2, len(historial) // 3)
    inicio = np.median(puntos[:k], axis=0)
    fin = np.median(puntos[-k:], axis=0)
    dt = float(np.median(tiempos[-k:]) - np.median(tiempos[:k]))
    if dt <= 1e-3:
        return 0.0
    return float(np.mean(np.linalg.norm(fin - inicio, axis=1)) / dt)

"""suavizar_adaptativo necesita reaccionar de inmediato, frame por frame. 
Si esperara una ventana de tiempo, las esquinas se quedarían atrás,
que es justo lo que quiere evitar. medir_movimiento_tiempo necesita ser confiable,
no rápida. Una decisión equivocada de "está quieta" produce una captura borrosa, 
así que conviene mirar varios frames y filtrar el ruido"""

# 7. P5 - DISTANCIA
def dimension_mayor(pts):
    """
    PROBLEMA : tarjeta lejos = pocos pixeles por caracter.
    SOLUCION : largo del lado mayor del POLIGONO de las 4 esquinas
    (no de la caja de YOLO: la caja crece con la mano y da una medida
    inflada). Si < ANCHO_MINIMO_TARJETA se muestra "Acerca mas".
    """
    tl, tr, br, bl = pts
    return max(_dist(tl, tr), _dist(bl, br), _dist(tl, bl), _dist(tr, br))


# 8. P6 - PERSPECTIVA, INCLINACION Y ROTACION
def _interseccion(l1, l2):
    """Punto donde se cruzan dos rectas dadas como (punto, direccion)."""
    (p1, d1), (p2, d2) = l1, l2
    A = np.array([d1, -d2], dtype=np.float64).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, (p2 - p1).astype(np.float64))
    return (p1 + t[0] * d1).astype("float32")

def _ancho_alto(q):
    """Ancho y alto promedio de un cuadrilatero [TL, TR, BR, BL]."""
    ancho = (_dist(q[0], q[1]) + _dist(q[3], q[2])) / 2
    alto = (_dist(q[0], q[3]) + _dist(q[1], q[2])) / 2
    return ancho, alto

def _det3(a, b, c):
    """Determinante de 3 vectores 3D = (a x b) . c, escrito a mano (rapido)."""
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))

def aspecto_real(q, f, centro):
    """
    PROBLEMA : con la camara inclinada la tarjeta se ve "aplastada": en tu
    escritorio su proporcion en la IMAGEN era ~2.0, aunque la credencial
    real mide 85.6 x 54 mm (1.586). Comparar contra 1.586 en pixeles hacia
    que el codigo "estirara" el recuadro hacia arriba (esquinas raras).
    SOLUCION : calcular la proporcion REAL del rectangulo visto en
    perspectiva (metodo de Zhang y He, 2006, usado en escaneres de
    pizarrones). Con los 4 vertices y la focal f se reconstruyen las dos
    direcciones del plano y se mide su razon.
    q = [TL, TR, BR, BL] en pixeles del frame completo.
    f puede ser un numero o un ARREGLO de focales: se evaluan todas de
    una vez (numpy), sin ciclo de Python.
    """
    tl, tr, br, bl = [(float(p[0]), float(p[1]), 1.0) for p in q]
    with np.errstate(divide="ignore", invalid="ignore"):
        k2 = _det3(tl, br, bl) / _det3(tr, br, bl)
        k3 = _det3(tl, br, tr) / _det3(bl, br, tr)
        n2 = np.array([k2 * tr[0] - tl[0], k2 * tr[1] - tl[1], k2 - 1.0])
        n3 = np.array([k3 * bl[0] - tl[0], k3 * bl[1] - tl[1], k3 - 1.0])
        f = np.asarray(f, dtype=np.float64)
        u0, v0 = centro

        def norma2(n):   # |K^-1 n|^2 con K = [[f,0,u0],[0,f,v0],[0,0,1]]
            return ((n[0] - u0 * n[2]) ** 2 + (n[1] - v0 * n[2]) ** 2) / f ** 2 + n[2] ** 2

        return np.sqrt(norma2(n2) / norma2(n3))

# Focales plausibles (se calculan una vez, no en cada llamada)
_FOCALES = np.linspace(FOCAL_REL[0], FOCAL_REL[1], 10)


def error_aspecto(q, shape):
    """
    Que tan lejos esta q de ser una credencial real (0 = perfecto).
    No conocemos la focal exacta de tu camara, asi que se prueba un rango
    plausible (FOCAL_REL) y se toma el MENOR error: "existe una camara
    razonable desde la que este cuadrilatero seria una tarjeta ID-1".
    """
    h, w = shape[:2]
    vals = aspecto_real(q, _FOCALES * w, (w / 2, h / 2))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 9.0
    return float(np.min(np.abs(vals - ASPECTO_ID1)) / ASPECTO_ID1)


def _alinear_con_yolo(q, pts_yolo):
    """
    El contorno devuelve las 4 esquinas en un orden arbitrario. Se prueban
    las 8 formas de recorrerlas (4 giros x 2 sentidos) y se elige la que
    mas se parece a los keypoints de YOLO: asi cada esquina refinada
    conserva el indice que le dio best.pt (0=TL, 1=TR, 2=BR, 3=BL).
    """
    mejor, mejor_d = q, 1e18
    for base in (q, q[::-1]):
        for k in range(4):
            cand = np.roll(base, -k, axis=0)
            d = float(np.sum(np.linalg.norm(cand - pts_yolo, axis=1)))
            if d < mejor_d:
                mejor, mejor_d = cand, d
    return mejor.astype("float32")


def segmentar_tarjeta(frame, pts_yolo, escala, iteraciones):
    """
    PASO A - separar la tarjeta del fondo (GrabCut, como en LearnOpenCV).
    PROBLEMA : los keypoints de YOLO solo son aproximados.
    SOLUCION : GrabCut modela con mezclas gaussianas los colores de
    "tarjeta" y de "fondo" y corta la imagen por el borde que mejor los
    separa. Se siembra con el poligono de YOLO:
      * 40 % central del poligono ...... seguro tarjeta
      * poligono de YOLO ................ probablemente tarjeta
      * hasta 160 % del poligono ........ probablemente fondo
      * resto ........................... seguro fondo
    Se trabaja en una ROI reducida (escala) para que sea rapido y luego
    una apertura/cierre morfologico quita dedos delgados y huecos.
    Retorna (mascara, roi_reducida, (x0, y0, escala)).
    """
    alto, ancho = frame.shape[:2]
    diag = _dist(pts_yolo[0], pts_yolo[2])
    mg = int(0.35 * diag)
    x0, y0 = np.maximum(pts_yolo.min(0) - mg, 0).astype(int)
    x1, y1 = np.minimum(pts_yolo.max(0) + mg, [ancho, alto]).astype(int)

    roi = cv2.resize(frame[y0:y1, x0:x1], None, fx=escala, fy=escala,
                     interpolation=cv2.INTER_AREA)
    P = (pts_yolo - [x0, y0]) * escala
    c = P.mean(axis=0)

    m = np.full(roi.shape[:2], cv2.GC_BGD, np.uint8)
    cv2.fillPoly(m, [((P - c) * 1.60 + c).astype(np.int32)], cv2.GC_PR_BGD)
    cv2.fillPoly(m, [((P - c) * 1.00 + c).astype(np.int32)], cv2.GC_PR_FGD)
    cv2.fillPoly(m, [((P - c) * 0.40 + c).astype(np.int32)], cv2.GC_FGD)

    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.setRNGSeed(0)                         # resultado repetible
    cv2.grabCut(roi, m, None, bgd, fgd, iteraciones, cv2.GC_INIT_WITH_MASK)
    mascara = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    k = max(3, int(0.03 * _dist(P[0], P[2]))) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, se)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, se)
    return mascara, roi, (x0, y0, escala)


def cuadrilatero_de_mascara(mascara):
    """
    PASO B - del contorno a 4 rectas (contorno + approxPolyDP, LearnOpenCV).
    PROBLEMA : la INE tiene esquinas REDONDEADAS; si se toman los
    vertices del contorno, la esquina queda "mordida" hacia adentro.
    SOLUCION :
      1. Contorno mas grande -> envolvente convexa -> approxPolyDP hasta
         obtener exactamente 4 vertices (forma aproximada).
      2. A cada lado se le ajusta una RECTA (cv2.fitLine, robusta Huber)
         usando solo el 70 % central del lado, lejos de las curvas.
      3. Las esquinas finales son la INTERSECCION de rectas vecinas: el
         vertice "virtual" donde la tarjeta seria cuadrada, que es justo
         lo que necesita la homografia.
    """
    cs, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    contorno = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(contorno)
    peri = cv2.arcLength(hull, True)

    aprox = None
    for eps in np.linspace(0.01, 0.12, 40):
        a = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a) == 4:
            aprox = a.reshape(4, 2).astype("float32")
            break
    if aprox is None:
        return None

    puntos = contorno.reshape(-1, 2).astype("float32")
    rectas = []
    for k in range(4):
        a, b = aprox[k], aprox[(k + 1) % 4]
        L = _dist(a, b)
        u = (b - a) / L
        n = np.array([-u[1], u[0]], dtype="float32")
        rel = puntos - a
        t, d = rel @ u, rel @ n
        sel = puntos[(t > 0.15 * L) & (t < 0.85 * L) & (np.abs(d) < 0.04 * L + 3)]
        if len(sel) < 10:
            rectas.append((a, u))
            continue
        vx, vy, x, y = cv2.fitLine(sel.reshape(-1, 1, 2), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        rectas.append((np.array([x, y], "float32"), np.array([vx, vy], "float32")))

    q = [_interseccion(rectas[(k - 1) % 4], rectas[k]) for k in range(4)]
    if any(v is None for v in q):
        return None
    return np.array(q, dtype="float32")


def _fuerza_borde(gx, gy, centro, u, L, n=90):
    """Gradiente medio perpendicular a una recta (que tan "borde" es)."""
    nn = np.array([-u[1], u[0]], dtype="float32")
    pts = centro + np.linspace(-0.40, 0.40, n)[:, None] * L * u
    h, w = gx.shape
    x = np.clip(pts[:, 0].round().astype(int), 0, w - 1)
    y = np.clip(pts[:, 1].round().astype(int), 0, h - 1)
    return float(np.mean(np.abs(gx[y, x] * nn[0] + gy[y, x] * nn[1])))


def ajustar_a_bordes(roi, q, rango_rel=RANGO_AJUSTE_BORDE):
    """
    PASO C - ajuste fino al borde (precision de pixel).
    PROBLEMA : GrabCut decide por COLOR; el limite de la mascara puede
    quedar 1-3 px corrido respecto al borde real.
    SOLUCION : cada lado se desplaza y gira un poco (+-0.8 % y +-1.5 grados)
    y se queda donde el gradiente perpendicular (Sobel) es maximo, es decir,
    exactamente sobre el escalon de intensidad del borde. El rango es
    pequeno a proposito para no "saltar" a lineas internas de la tarjeta
    (la franja de UNITED STATES, la banda inferior) ni a letras del fondo.
    Retorna (cuadrilatero, fuerza_de_cada_lado).
    """
    g = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    ancho, alto = _ancho_alto(q)
    rectas, fuerza = [], []
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        L = _dist(a, b)
        u = (b - a) / L
        n = np.array([-u[1], u[0]], dtype="float32")
        medio = (a + b) / 2
        r = rango_rel * (alto if k % 2 == 0 else ancho) + 2

        mejor = (-1.0, medio, u)
        for dth in np.radians(np.arange(-1.5, 1.51, 0.5)):
            c, s = np.cos(dth), np.sin(dth)
            u2 = np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]], dtype="float32")
            for d in np.arange(-r, r + 0.01, 0.5):
                m = medio + n * d
                f = _fuerza_borde(gx, gy, m, u2, L)
                if f > mejor[0]:
                    mejor = (f, m, u2)
        rectas.append((mejor[1], mejor[2]))
        fuerza.append(mejor[0])

    q2 = [_interseccion(rectas[(k - 1) % 4], rectas[k]) for k in range(4)]
    if any(v is None for v in q2):
        return q, fuerza
    return np.array(q2, dtype="float32"), fuerza


def corregir_lado_debil(q, fuerza, shape=None, escala=1.0, offset=(0, 0), tol=TOL_ASPECTO):
    """
    PASO D - usar la forma conocida de la credencial (85.6 x 54 mm).
    PROBLEMA : con poco contraste (INE blanca sobre papel blanco) un lado
    puede no verse y GrabCut lo corta en una linea interna (p. ej. la
    franja de "UNITED STATES OF AMERICA"): la tarjeta sale "aplastada".
    SOLUCION : si la proporcion ancho/alto se aleja mas de 'tol' de 1.586,
      * demasiado ancha  -> sobra ancho / falta alto: el culpable es el
        lado superior o inferior con MENOR fuerza de borde;
      * demasiado alta   -> el culpable es el izquierdo o derecho mas debil.
    Ese lado se recoloca sobre los lados confiables a la distancia que
    exige la proporcion ID-1. Retorna (cuadrilatero, se_corrigio).
    IMPORTANTE: solo se corrige si NINGUNA camara plausible explica la
    forma (error_aspecto). Una tarjeta vista inclinada se ve mas ancha en
    la imagen y eso NO es un lado perdido: antes se "corregia" y el
    recuadro se estiraba hacia arriba.
    """
    ancho, alto = _ancho_alto(q)
    asp = ancho / alto
    if shape is not None:
        if error_aspecto(q / escala + np.asarray(offset, "float32"), shape) <= tol:
            return q, False
    elif abs(asp - ASPECTO_ID1) / ASPECTO_ID1 <= tol:
        return q, False

    q = q.copy()
    if asp > ASPECTO_ID1:                       # falta alto
        objetivo = ancho / ASPECTO_ID1
        if fuerza[0] < fuerza[2]:               # superior debil: se mueve desde abajo
            q[0] = q[3] + (q[0] - q[3]) / _dist(q[3], q[0]) * objetivo
            q[1] = q[2] + (q[1] - q[2]) / _dist(q[2], q[1]) * objetivo
        else:                                   # inferior debil: se mueve desde arriba
            q[3] = q[0] + (q[3] - q[0]) / _dist(q[0], q[3]) * objetivo
            q[2] = q[1] + (q[2] - q[1]) / _dist(q[1], q[2]) * objetivo
    else:                                       # falta ancho
        objetivo = alto * ASPECTO_ID1
        if fuerza[1] < fuerza[3]:               # derecho debil
            q[1] = q[0] + (q[1] - q[0]) / _dist(q[0], q[1]) * objetivo
            q[2] = q[3] + (q[2] - q[3]) / _dist(q[3], q[2]) * objetivo
        else:                                   # izquierdo debil
            q[0] = q[1] + (q[0] - q[1]) / _dist(q[1], q[0]) * objetivo
            q[3] = q[2] + (q[3] - q[2]) / _dist(q[2], q[3]) * objetivo
    return q, True


def validar_cuadrilatero(q, pts_yolo, shape):
    """
    PASO E - red de seguridad: nunca empeorar a YOLO.
    El resultado se descarta (y se usan los keypoints de YOLO) si:
      * no es convexo,
      * ninguna camara plausible la ve como credencial (error_aspecto
        mayor a 25 %),
      * su area es menos de la mitad o mas del doble de la de YOLO,
      * alguna esquina se movio mas del 35 % de la diagonal.
    """
    if not cv2.isContourConvex(q.reshape(-1, 1, 2)):
        return False
    if error_aspecto(q, shape) > 0.25:
        return False
    a_y, a_q = cv2.contourArea(pts_yolo), cv2.contourArea(q)
    if a_y <= 0 or not (0.5 < a_q / a_y < 2.0):
        return False
    diag = _dist(pts_yolo[0], pts_yolo[2])
    return bool(np.all(np.linalg.norm(q - pts_yolo, axis=1) < 0.35 * diag))


def _mapas_bordes(roi):
    """Canny (con CLAHE para bordes tenues), Sobel y color Lab de la ROI."""
    g = _CLAHE.apply(
        cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    g = cv2.GaussianBlur(g, (5, 5), 0)
    med = float(np.median(g))
    bordes = cv2.Canny(g, max(8, 0.25 * med), max(25, 0.7 * med))
    gf = g.astype(np.float32)
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1)
    lab = cv2.cvtColor(cv2.GaussianBlur(roi, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    return bordes, cv2.dilate(bordes, np.ones((3, 3), np.uint8)), gx, gy, lab


def _calidad_lado(mapas, a, b, n=50):
    """
    Evidencia de que el segmento a-b es un borde REAL de la tarjeta (0-1):
      * cobertura : % del lado con pixeles de borde (Canny) orientados
                    como el lado;
      * contraste : diferencia de color (Lab) entre una franja justo
                    ADENTRO y otra justo AFUERA. Una linea dibujada en el
                    papel tiene papel de los dos lados (contraste ~0); el
                    borde de la credencial separa tarjeta de fondo.
    Se ignora el 15 % de cada extremo (esquinas redondeadas).
    """
    _, dil, gx, gy, lab = mapas
    v = b - a
    L = float(np.linalg.norm(v))
    if L < 10:
        return 0.0
    u = v / L
    nn = np.array([-u[1], u[0]], dtype="float32")
    p = a + np.linspace(0.15, 0.85, n)[:, None] * v
    h, w = dil.shape

    def muestrear(img, desplaz):
        q = p + nn * desplaz
        x = np.clip(q[:, 0].round().astype(int), 0, w - 1)
        y = np.clip(q[:, 1].round().astype(int), 0, h - 1)
        return img[y, x]

    ang = np.arctan2(muestrear(gy, 0), muestrear(gx, 0))
    dif = np.abs(np.angle(np.exp(1j * (ang - np.arctan2(nn[1], nn[0])))))
    dif = np.minimum(dif, np.pi - dif)
    cobertura = float(np.mean((muestrear(dil, 0) > 0) & (dif < np.radians(30))))
    contraste = float(np.linalg.norm(np.median(muestrear(lab, 5), axis=0)
                                     - np.median(muestrear(lab, -5), axis=0)))
    return 0.5 * cobertura + 0.5 * min(1.0, contraste / 12.0)


def puntuar_cuadrilatero(q, mapas, area_yolo, escala, offset, shape):
    """
    Puntaje de un candidato (mas alto = mas creible):
      + calidad de sus lados (los 3 mejores pesan completo y el peor 40 %:
        un lado puede verse poco, p. ej. INE blanca sobre papel blanco);
      - error de proporcion REAL respecto a 1.586;
      + un poco por tamano (prefiere el borde EXTERIOR de la tarjeta).
    Devuelve -9 si el candidato es imposible.
    """
    if q is None or not np.all(np.isfinite(q)):
        return -9.0
    if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype("float32")):
        return -9.0
    ar = cv2.contourArea(q.astype("float32")) / area_yolo
    if not (0.3 < ar < 1.6):
        return -9.0
    ea = error_aspecto(q / escala + offset, shape)
    if ea > 0.25:
        return -9.0
    lados = sorted((_calidad_lado(mapas, q[k], q[(k + 1) % 4]) for k in range(4)), reverse=True)
    return float(np.mean(lados[:3]) + 0.4 * lados[3] - 2.0 * ea + 0.25 * min(ar, 1.0))


def candidatos_por_lineas(mapas, P, max_lineas=7):
    """
    HIPOTESIS ALTERNATIVAS (no dependen del color):
    se buscan segmentos rectos (HoughLinesP sobre Canny), se separan en
    "horizontales" y "verticales" respecto a la orientacion que da YOLO y
    se combinan 2 + 2 para formar cuadrilateros. Sirve justo cuando
    GrabCut falla: tarjeta blanca sobre papel blanco (mismo color).
    """
    bordes = mapas[0]
    ux = (P[1] - P[0]) / (np.linalg.norm(P[1] - P[0]) + 1e-9)
    uy = np.array([-ux[1], ux[0]], dtype="float32")
    ref = np.degrees(np.arctan2(ux[1], ux[0]))
    _, alto = _ancho_alto(P)

    segs = cv2.HoughLinesP(bordes, 1, np.pi / 360, threshold=30,
                           minLineLength=max(10, int(0.20 * alto)), maxLineGap=8)
    horiz, vert = [], []
    if segs is not None:
        for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
            a = np.array([x1, y1], "float32")
            b = np.array([x2, y2], "float32")
            L = float(np.linalg.norm(b - a))
            u = (b - a) / L
            rel = ((np.degrees(np.arctan2(u[1], u[0])) - ref + 90) % 180) - 90
            if abs(rel) < 25:
                horiz.append((L, (a + b) / 2, u if np.dot(u, ux) >= 0 else -u))
            elif abs(abs(rel) - 90) < 25:
                vert.append((L, (a + b) / 2, u if np.dot(u, uy) >= 0 else -u))

    def sin_repetir(lista):
        out = []
        for L, m, u in sorted(lista, key=lambda r: -r[0]):
            n = np.array([-u[1], u[0]])
            if all(abs(np.dot(m - m2, n)) > 6 or abs(np.dot(u, u2)) < np.cos(np.radians(4))
                   for _, m2, u2 in out):
                out.append((L, m, u))
            if len(out) >= max_lineas:
                break
        return [(m, u) for _, m, u in out]

    horiz, vert = sin_repetir(horiz), sin_repetir(vert)
    cands = []
    for h1, h2 in itertools.combinations(horiz, 2):
        sup, inf = (h1, h2) if np.dot(h1[0] - h2[0], uy) < 0 else (h2, h1)
        for v1, v2 in itertools.combinations(vert, 2):
            izq, der = (v1, v2) if np.dot(v1[0] - v2[0], ux) < 0 else (v2, v1)
            rectas = [sup, der, inf, izq]
            q = [_interseccion(rectas[(k - 1) % 4], rectas[k]) for k in range(4)]
            if all(v is not None for v in q):
                cands.append(np.array(q, dtype="float32"))
    return cands


def refinar_esquinas(frame, pts_yolo, preciso=False):
    """
    PROBLEMA : los keypoints de YOLO no caen exactamente en las esquinas
    (con la tarjeta de lado pueden quedar 50-150 px fuera) y la
    homografia sale chueca.
    SOLUCION : generar VARIAS hipotesis de las 4 esquinas y quedarse con
    la que mejor explica la imagen:
      1. GrabCut sembrado por YOLO -> contorno -> 4 rectas -> ajuste fino
         (funciona muy bien si la tarjeta contrasta con el fondo);
      2. la misma, con el lado debil corregido por la proporcion ID-1;
      3. cuadrilateros formados con lineas rectas detectadas (Hough),
         utiles si el fondo tiene el mismo color que la tarjeta.
    Cada hipotesis se califica con puntuar_cuadrilatero (bordes reales +
    contraste dentro/fuera + proporcion REAL 1.586). Entre candidatos casi
    empatados se prefiere el mas grande (el borde exterior de la tarjeta,
    no una franja interna). Al final, validar_cuadrilatero evita empeorar
    a YOLO.
    preciso=False : version rapida para la vista en vivo.
    preciso=True  : version fina, solo para el frame que se guarda.
    Retorna (esquinas [TL,TR,BR,BL], metodo).
    """
    pts_yolo = np.asarray(pts_yolo, dtype="float32")
    if not REFINAR_ESQUINAS:
        return pts_yolo, "YOLO"

    if preciso:
        escala, iters = ESCALA_FINAL, ITER_FINAL
    else:
        # GrabCut es lo mas caro del ciclo y su costo crece con los pixeles.
        # En vivo basta con que la tarjeta mida ~DIAG_VIVO px: si esta cerca
        # de la camara (se ve grande) se reduce mas, si esta lejos, menos.
        escala = min(ESCALA_VIVO, DIAG_VIVO / max(1.0, _dist(pts_yolo[0], pts_yolo[2])))
        iters = ITER_VIVO
    try:
        mascara, roi, (x0, y0, s) = segmentar_tarjeta(frame, pts_yolo, escala, iters)
        offset = np.array([x0, y0], dtype="float32")
        P = ((pts_yolo - offset) * s).astype("float32")
        mapas = _mapas_bordes(roi)
        area_yolo = max(1.0, cv2.contourArea(P))

        candidatos = []                                   # (nombre, cuadrilatero en la ROI)
        q = cuadrilatero_de_mascara(mascara)
        if q is not None:
            q = _alinear_con_yolo(q, P)
            q, fuerza = ajustar_a_bordes(roi, q)
            candidatos.append(("REFINADAS", q))
            q2, corregido = corregir_lado_debil(q, fuerza, frame.shape, s, offset)
            if corregido:
                candidatos.append(("REFINADAS +ID1", q2))
        # (en vivo se usan menos lineas: 5x5 combinaciones en vez de 7x7)
        for c in candidatos_por_lineas(mapas, P, max_lineas=7 if preciso else 5):
            candidatos.append(("REFINADAS LINEAS", _alinear_con_yolo(c, P)))

        puntuados = [(puntuar_cuadrilatero(c, mapas, area_yolo, s, offset, frame.shape), nom, c)
                     for nom, c in candidatos]
        puntuados = [t for t in puntuados if t[0] > -9]
        if not puntuados:
            return pts_yolo, "YOLO"

        mejor_puntaje = max(t[0] for t in puntuados)
        empatados = [t for t in puntuados if t[0] >= mejor_puntaje - 0.03]
        _, metodo, q = max(empatados, key=lambda t: cv2.contourArea(t[2].astype("float32")))
        q = (q / s + offset).astype("float32")
    except (cv2.error, ValueError, ZeroDivisionError):
        return pts_yolo, "YOLO"

    if not validar_cuadrilatero(q, pts_yolo, frame.shape):
        return pts_yolo, "YOLO"
    return q, metodo


def evaluar_geometria(pts):
    """
    PROBLEMA : "NO SIEMPRE se debe corregir": warpear una tarjeta que ya
    esta recta solo la re-muestrea y pierde nitidez.
    SOLUCION : medir que tan recta esta la tarjeta con 3 criterios:
      * inclinacion : bordes vs horizontal/vertical (rotacion en el plano)
      * esquinas    : desvio respecto a 90 grados (cizalla / perspectiva)
      * distorsion  : diferencia entre lados opuestos (la tarjeta esta
                      girada hacia la camara y un lado se ve mas corto)
    Si TODOS estan dentro de tolerancia -> recta -> no hay homografia.
    Reemplaza al viejo ratio area_puntos/area_caja, que con la mano
    dentro de la caja caia a 0.53 y mandaba siempre a "BOUNDING BOX".
    Supone el orden [TL, TR, BR, BL] (lo garantiza verificar_labels.py).
    """
    tl, tr, br, bl = pts

    ang_sup = _angulo(tl, tr)                 # 0 = horizontal
    ang_inf = _angulo(bl, br)
    dev_izq = _angulo(tl, bl) - 90.0          # 0 = vertical
    dev_der = _angulo(tr, br) - 90.0
    inclinacion = max(abs(ang_sup), abs(ang_inf), abs(dev_izq), abs(dev_der))

    esquinas = [
        _angulo_interior(bl, tl, tr),
        _angulo_interior(tl, tr, br),
        _angulo_interior(tr, br, bl),
        _angulo_interior(br, bl, tl),
    ]
    dev_esquina = max(abs(a - 90.0) for a in esquinas)

    ls, li = _dist(tl, tr), _dist(bl, br)
    lz, ld = _dist(tl, bl), _dist(tr, br)
    distorsion = max(abs(ls - li) / max(ls, li), abs(lz - ld) / max(lz, ld))

    motivos = []
    if inclinacion > TOL_INCLINACION_GRADOS:
        motivos.append("inclinada")
    if dev_esquina > TOL_ESQUINA_GRADOS:
        motivos.append("esquinas")
    if distorsion > TOL_RAZON_LADOS:
        motivos.append("perspectiva")

    return {
        "ang_sup": ang_sup,
        "inclinacion": inclinacion,
        "dev_esquina": dev_esquina,
        "distorsion": distorsion,
        "necesita_homografia": len(motivos) > 0,
        "motivo": "+".join(motivos) if motivos else "recta",
    }


def rectificar(frame, pts, geo, w=ANCHO_DESTINO, h=ALTO_DESTINO):
    """
    PROBLEMA : perspectiva, inclinacion y rotacion deforman el texto.
    SOLUCION :
      * necesita_homografia -> cv2.getPerspectiveTransform +
        warpPerspective: las 4 esquinas pasan a ser un rectangulo
        perfecto de 856x540 (vista frontal). Interpolacion bicubica y
        BORDER_REPLICATE para no meter bordes negros.
      * ya esta recta -> solo se recorta el rectangulo (sin re-muestreo
        por homografia) y se escala al tamano estandar.
    Retorna (imagen, metodo).
    """
    if geo["necesita_homografia"]:
        dst = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32"
        )
        M = cv2.getPerspectiveTransform(pts.astype("float32"), dst)
        out = cv2.warpPerspective(
            frame, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return out, "HOMOGRAFIA"

    alto, ancho = frame.shape[:2]
    x1 = max(0, int(np.floor(pts[:, 0].min())))
    y1 = max(0, int(np.floor(pts[:, 1].min())))
    x2 = min(ancho, int(np.ceil(pts[:, 0].max())))
    y2 = min(alto, int(np.ceil(pts[:, 1].max())))

    out = cv2.resize(frame[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_CUBIC)
    return out, "RECORTE RECTO"


# 9. P7 - BAJO CONTRASTE Y RUIDO
def mejorar_imagen(img, baja_luz=False, final=False):
    """
    Cadena de mejora. Dos modos:
      final=False -> vista previa rapida (solo CLAHE).
      final=True  -> imagen que se manda a Ollama (cadena completa).

    Orden (importa):
      1. [P1] Denoise fastNlMeans (h=6) solo si hubo poca luz: con poca
         luz la camara sube la ganancia y aparece ruido; se limpia ANTES
         de realzar contraste para no amplificarlo.
      2. [P7] CLAHE en el canal L de LAB (clip 2.0, 8x8): contraste LOCAL
         sin cambiar los colores (trabajar en L evita tintes raros).
      3. [P7] Filtro bilateral (9,75,75): suaviza ruido y grano SIN
         borrar los bordes de las letras.
      4. [P3] Unsharp mask (1.4 / -0.4): realza bordes al final, para
         recuperar la nitidez que quitaron los pasos anteriores.
    """
    if final and baja_luz:
        img = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)

    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = _CLAHE.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    if not final:
        return out

    out = cv2.bilateralFilter(out, 9, 75, 75)
    out = cv2.addWeighted(out, 1.4, cv2.GaussianBlur(out, (0, 0), 1.5), -0.4, 0)
    return out

#_____________________________________________________________________________________________

# 10. DIBUJO Y GUARDADO DE 0_deteccion.jpg
COLORES_PTS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
NOMBRES_PTS = ["0 - TL", "1 - TR", "2 - BR", "3 - BL"]

def dibujar_poligono(imagen, pts, grosor=2, radio=8, etiquetas=True):
    """Poligono amarillo que une los 4 puntos (sin bounding box)."""
    cv2.polylines(imagen, [pts.astype(int).reshape(-1, 1, 2)], True,
                  (0, 255, 255), grosor)
    for i, (x, y) in enumerate(pts):
        px, py = int(x), int(y)
        cv2.circle(imagen, (px, py), radio, COLORES_PTS[i], -1)
        if etiquetas:
            cv2.putText(imagen, NOMBRES_PTS[i], (px + 12, py - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORES_PTS[i], 2)


def guardar_deteccion_original(frame, pts, ruta, datos):
    """
    Guarda el cuadro original + poligono amarillo + panel con TODOS los
    parametros de la captura (sirve para depurar por que se acepto o
    rechazo un frame).
    """
    imagen = frame.copy()
    dibujar_poligono(imagen, pts, grosor=3, radio=9)

    alto = imagen.shape[0]
    ANCHO_PANEL = 480
    esc = alto / 1080.0

    panel = np.full((alto, ANCHO_PANEL, 3), 35, dtype=np.uint8)
    cv2.putText(panel, "DIAGNOSTICO DEL ESCANEO", (20, int(45 * esc) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    cv2.line(panel, (20, int(65 * esc) + 10), (ANCHO_PANEL - 20, int(65 * esc) + 10),
             (100, 100, 100), 2)

    lineas = [
        ("Confianza YOLO", f"{datos['conf']:.3f}"),
        ("Nitidez Laplacian", f"{datos['nit']:.2f}"),
        ("Reflejos", f"{datos['ref']:.2f}%"),
        ("Iluminacion", f"{datos['brillo']:.2f}"),
        ("Movimiento", f"{datos['mov']:.0f} px/s"),
        ("Dimension mayor", f"{datos['ancho']} px"),
        ("Angulo superior", f"{datos['angulo']:.2f} grados"),
        ("Inclinacion max", f"{datos['inclinacion']:.2f} grados"),
        ("Distorsion lados", f"{datos['distorsion'] * 100:.1f}%"),
        ("Esquinas", datos["esquinas"]),
        ("Correccion", datos["metodo"]),
        ("Gamma", "ACTIVADO" if datos["gamma"] else "DESACTIVADO"),
        ("Estado", datos["estado"]),
    ]

    y0 = int(110 * esc) + 10
    paso = min(70, (alto - y0 - 20) // len(lineas))

    for i, (nombre, valor) in enumerate(lineas):
        y = y0 + i * paso
        cv2.putText(panel, nombre + ":", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1)
        cv2.putText(panel, str(valor), (235, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 100), 2)

    cv2.imwrite(ruta, np.hstack((imagen, panel)))


# 11. PIPELINE POR FRAME  (sin camara: facil de probar)
def procesar_frame(work, pts):
    """
    Aplica la cadena de analisis a un frame (ya con gamma si hacia falta):
    geometria -> rectificacion (solo si hace falta) -> metricas de calidad.
    'pts' son las 4 esquinas YA REFINADAS, en orden TL,TR,BR,BL.
    """
    geo = evaluar_geometria(pts)
    persp, metodo = rectificar(work, pts, geo)

    return {
        "persp": persp,
        "metodo": metodo,
        "geo": geo,
        "nit": medir_nitidez(persp),
        "ref": medir_reflejos(persp),
        "ancho": int(dimension_mayor(pts)),
    }

# 11b. CARPETAS DE CAPTURAS
CARPETA_CAPTURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capturas")
ABRIR_CARPETA = True            # al terminar cada escaneo, abre su carpeta


def abrir_carpeta(carpeta):
    """
    Abre la carpeta del escaneo con el explorador de archivos del sistema:
      * Mac     -> Finder          (comando "open")
      * Windows -> Explorador      (os.startfile)
      * Linux   -> el que tenga    (comando "xdg-open")
    Si falla (p. ej. un servidor sin pantalla), no pasa nada: las imagenes
    ya estan guardadas.
    """
    if not ABRIR_CARPETA:
        return
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", carpeta])
        elif sys.platform.startswith("win"):
            os.startfile(carpeta)
        else:
            subprocess.Popen(["xdg-open", carpeta])
    except (OSError, AttributeError):
        pass

def nueva_carpeta_captura(base=None):
    """
    Crea capturas/imagen_1, imagen_2, imagen_3... (la siguiente libre).
    Cada escaneo guarda ahi sus 4 imagenes, asi no se sobrescriben entre si.
    """
    base = base or CARPETA_CAPTURAS
    os.makedirs(base, exist_ok=True)
    usados = []
    for nombre in os.listdir(base):
        partes = nombre.split("_")
        if len(partes) == 2 and partes[0] == "imagen" and partes[1].isdigit():
            usados.append(int(partes[1]))
    carpeta = os.path.join(base, f"imagen_{max(usados, default=0) + 1}")
    os.makedirs(carpeta)
    return carpeta

# 12. CAPTURA PRINCIPAL
def aplicar_perfil_camara(camara):
    """
    Ajusta los umbrales segun la camara (ver PERFILES_CAMARA). Si la
    camara no tiene perfil, se usa PERFIL_GENERICO (promedio de ambas).
    """
    perfil = PERFILES_CAMARA.get(camara, PERFIL_GENERICO)
    for nombre, valor in perfil.items():
        if nombre not in globals():
            raise KeyError(f"El perfil de la camara {camara} usa '{nombre}', "
                           "que no es un parametro de este archivo")
        globals()[nombre] = valor
    nombre_cam = NOMBRES_CAMARA.get(camara, "Camara desconocida")
    tipo = "perfil propio" if camara in PERFILES_CAMARA else "perfil generico (promedio webcam/iPhone)"
    print(f"\nCamara {camara} - {nombre_cam}: {tipo}")
    for nombre in perfil:
        print(f"  {nombre:22s} = {globals()[nombre]}")


_CAMARA_ELEGIDA = None          # se pregunta una sola vez por sesion


def elegir_camara():
    """
    Pregunta al usuario que camara usara (0 webcam, 1 generica, 2 iPhone).
    ENTER usa la camara por defecto (CAMARA). Se pregunta SOLO la primera
    vez: los siguientes escaneos de la misma sesion reutilizan la eleccion.
    """
    global _CAMARA_ELEGIDA
    if _CAMARA_ELEGIDA is not None:
        return _CAMARA_ELEGIDA

    print("\n=== SELECCION DE CAMARA ===")
    for num, nombre in NOMBRES_CAMARA.items():
        marca = "  (por defecto)" if num == CAMARA else ""
        print(f"  {num} - {nombre}{marca}")

    while True:
        opcion = input("Elige la camara [0/1/2, ENTER = por defecto]: ").strip()
        if opcion == "":
            _CAMARA_ELEGIDA = CAMARA
            break
        if opcion.isdigit() and int(opcion) in NOMBRES_CAMARA:
            _CAMARA_ELEGIDA = int(opcion)
            break
        print("Opcion no valida, escribe 0, 1 o 2.")
    return _CAMARA_ELEGIDA


def capturar_y_procesar(ruta_modelo="best.pt", camara=None):
    from ultralytics import YOLO     # import local: lo demas no lo necesita

    model = YOLO(ruta_modelo)

    camara = elegir_camara() if camara is None else camara
    aplicar_perfil_camara(camara)
    cap = cv2.VideoCapture(camara)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    lut = crear_tabla_gamma()

    st = {
        "perdidos": 0, "nit": -1.0,
        "t_inicio": None,       # desde cuando la tarjeta esta valida
        "t_valido": None,       # ultimo instante con frame valido
        "hist": deque(),        # (tiempo, keypoints crudos) para el movimiento
        "best": None,           # mejor frame valido de la secuencia
        "pts": None,            # keypoints de YOLO suavizados (EMA)
        "corr": None,           # correccion YOLO -> esquina real (por esquina)
        "metodo_esq": "YOLO",   # como se obtuvieron las esquinas
        "frame_n": 0,           # contador para refinar cada REFINAR_CADA
    }
    ruta_final = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- P1: poca iluminacion ----
        brillo = medir_brillo(frame)
        baja_luz = brillo < UMBRAL_BAJA_LUZ
        work = cv2.LUT(frame, lut) if baja_luz else frame
        vis = work.copy()

        candidato = None
        detectado = False
        msg, col = "Esperando credencial...", (0, 0, 255)

        cv2.putText(vis, f"Luz:{int(brillo)} {'[GAMMA]' if baja_luz else ''}",
                    (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

        # Histeresis de confianza (estabilidad): para EMPEZAR a seguir la
        # tarjeta se exige CONF_YOLO, pero si ya se estaba siguiendo basta
        # CONF_YOLO_SEGUIR. Asi un frame dudoso (tarjeta de lado, reflejo,
        # mano encima) no apaga el recuadro amarillo.
        umbral_conf = CONF_YOLO if st["pts"] is None else CONF_YOLO_SEGUIR

        # max_det=1: solo interesa la tarjeta mas segura (menos trabajo en NMS)
        for r in model(work, conf=CONF_YOLO_SEGUIR, device=DEVICE,
                       max_det=1, verbose=False):

            if not (r.boxes is not None and len(r.boxes) > 0
                    and r.keypoints is not None and len(r.keypoints.xy) > 0):
                continue

            raw = r.keypoints.xy[0].cpu().numpy()
            if len(raw) != 4 or not np.all(np.isfinite(raw)) or not np.all(raw > 0):
                continue

            conf = float(r.boxes.conf[0].cpu().numpy())
            if conf < umbral_conf:
                continue

            detectado = True
            st["perdidos"] = 0

            # P4: suavizado EMA de los keypoints de YOLO (para que el recuadro amarillo no tiemble)
            st["pts"] = suavizar_adaptativo(raw, st["pts"])

            # ---- P6: esquinas exactas a partir del contorno real ----
            # GrabCut es lo mas caro del ciclo (~0.1 s). Correrlo en TODOS
            # los frames bajaba la camara a pocos FPS: la tarjeta tardaba
            # en detectarse y el contador "Capturando x/10" avanzaba lento.
            # Ahora se refina 1 de cada REFINAR_CADA frames y se guarda la
            # CORRECCION (esquina real - keypoint YOLO) de cada esquina.
            # En los frames intermedios: esquinas = YOLO + correccion, que
            # cuesta una suma. Como YOLO sigue a la tarjeta en cada frame,
            # el recuadro amarillo la acompana aunque se mueva.
            # (Al guardar se vuelve a refinar en modo preciso, sin atajos.)
            st["frame_n"] += 1
            if st["corr"] is None or st["frame_n"] % REFINAR_CADA == 0:
                esq, metodo = refinar_esquinas(work, st["pts"], preciso=False)
                if metodo != "YOLO":
                    # La nueva correccion se MEZCLA con la anterior (EMA) en
                    # vez de reemplazarla: el recuadro se desliza, no salta.
                    # (si antes no habia correccion valida, se toma directa)
                    previa = st["corr"] if st["metodo_esq"] != "YOLO" else None
                    st["corr"] = suavizar_ema(esq - st["pts"], previa,
                                              alpha=ALPHA_CORRECCION)
                    st["metodo_esq"] = metodo
                elif st["corr"] is None:
                    st["corr"] = np.zeros((4, 2), dtype="float32")
                # si el refinado fallo, se conserva la correccion anterior
            pts = (st["pts"] + st["corr"]).astype("float32")
            metodo_esq = st["metodo_esq"]

            # ---- P6 + metricas de P2, P3, P4, P5 ----
            res = procesar_frame(work, pts)
            # ---- P4: movimiento REAL en px/s, robusto al ruido de YOLO ----
            ahora = time.monotonic()
            st["hist"].append((ahora, raw.astype("float32").copy()))
            while st["hist"] and ahora - st["hist"][0][0] > VENTANA_MOVIMIENTO:
                st["hist"].popleft()
            res["mov"] = medir_movimiento_tiempo(st["hist"])
            geo = res["geo"]

            # ---- Validacion: el primer problema que falle define el mensaje ----
            valida = False
            if res["ancho"] < ANCHO_MINIMO_TARJETA:                  # P5
                msg, col = "Acerca mas...", (0, 165, 255)
            elif res["mov"] > UMBRAL_MOVIMIENTO_MAX:                 # P4
                msg, col = "Mantenla quieta...", (0, 165, 255)
            elif res["ref"] > UMBRAL_REFLEJO_MAX:                    # P2
                msg, col = "Reflejo - inclina...", (0, 0, 255)
            elif res["nit"] < UMBRAL_NITIDEZ_MIN:                    # P3
                msg, col = "Enfocando...", (0, 255, 255)
            else:
                # ---- Cuenta por TIEMPO, no por numero de frames ----
                # Antes se pedian 10 frames seguidos: a 60 FPS eran 0.17 s
                # (demasiado poco) y un solo frame ruidoso reiniciaba todo.
                # Ahora la tarjeta debe estar valida SEGUNDOS_PARA_CAPTURA
                # segundos, sin importar cuantos FPS de la camara.
                valida = True
                st["t_valido"] = ahora
                if st["t_inicio"] is None:
                    st["t_inicio"] = ahora
                avance = min(1.0, (ahora - st["t_inicio"]) / SEGUNDOS_PARA_CAPTURA)
                msg = f"Capturando {int(avance * 100)}%"
                col = (0, 255, 0)

            datos = {
                "conf": conf, "nit": res["nit"], "ref": res["ref"],
                "brillo": brillo, "mov": res["mov"], "ancho": res["ancho"],
                "angulo": geo["ang_sup"], "inclinacion": geo["inclinacion"],
                "distorsion": geo["distorsion"], "metodo": res["metodo"],
                "esquinas": metodo_esq,
                "gamma": baja_luz,
                "estado": "CAPTURA VALIDA" if valida else "NO VALIDA",
            }

            # ---- Vista en vivo ----
            dibujar_poligono(vis, pts, grosor=2, radio=7, etiquetas=False)
            cv2.imshow("P2 - Limpia", mejorar_imagen(res["persp"]))
            cv2.putText(
                vis,
                f"Conf:{conf:.2f} | Nit:{int(res['nit'])} | Ref:{res['ref']:.1f}% | "
                f"Anc:{res['ancho']} | Mov:{res['mov']:.0f}px/s | {res['metodo']}",
                (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
            )
            # camino que gano en el ultimo refinado en vivo
            cv2.putText(vis, f"Esquinas: {metodo_esq}", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            candidato = {
                # sin .copy(): cap.read() entrega un arreglo NUEVO en cada
                # vuelta y estos no se modifican despues (se dibuja en 'vis')
                "orig": frame, "work": work,
                "yolo": st["pts"].copy(), "persp": res["persp"],
                "pts": pts.copy(), "datos": datos,
                "luz": baja_luz, "nit": res["nit"],
            }

            # ---- P3: conservar el frame valido MAS nitido ----
            if valida and res["nit"] > st["nit"]:
                st["nit"] = res["nit"]
                st["best"] = candidato

        # ---- Deteccion perdida: tolerancia antes de soltar la tarjeta ----
        # Si YOLO falla unos pocos frames, el recuadro se sigue dibujando en
        # su ultima posicion y el contador de captura se PAUSA (no vuelve a
        # 0). Solo tras FRAMES_TOLERANCIA frames seguidos sin tarjeta se
        # reinicia todo.
        if not detectado:
            st["perdidos"] += 1
            if st["pts"] is not None and st["perdidos"] <= FRAMES_TOLERANCIA:
                ultimo = (st["pts"] + st["corr"]).astype("float32")
                dibujar_poligono(vis, ultimo, grosor=2, radio=7, etiquetas=False)
                msg, col = "Buscando credencial...", (0, 200, 255)
            else:
                st.update({"t_inicio": None, "t_valido": None, "nit": -1.0,
                           "best": None, "pts": None, "corr": None,
                           "metodo_esq": "YOLO"})
                st["hist"].clear()
        elif candidato is not None and candidato["datos"]["estado"] != "CAPTURA VALIDA":
            # Un frame malo suelto (ruido, parpadeo) NO reinicia la cuenta;
            # solo si la tarjeta lleva mas de SEGUNDOS_TOLERANCIA mal.
            if st["t_valido"] is None or time.monotonic() - st["t_valido"] > SEGUNDOS_TOLERANCIA:
                st.update({"t_inicio": None, "t_valido": None,
                           "nit": -1.0, "best": None})

        cv2.putText(vis, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cv2.imshow("P1 - Camara", vis)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break

        # ---- Guardado: automatico (SEGUNDOS_PARA_CAPTURA) o manual con 's' ----
        automatico = (st["t_inicio"] is not None and st["best"] is not None
                      and time.monotonic() - st["t_inicio"] >= SEGUNDOS_PARA_CAPTURA)
        manual = k == ord("s") and candidato is not None

        if automatico or manual:
            sel = st["best"] if automatico else candidato

            # ---- P6: refinamiento PRECISO solo para el frame elegido ----
            # (en vivo se usa la version rapida; aqui se repite con mas
            #  resolucion e iteraciones, y sin el retraso del EMA)
            esq, metodo_esq = refinar_esquinas(sel["work"], sel["yolo"], preciso=True)
            geo = evaluar_geometria(esq)
            persp, metodo = rectificar(sel["work"], esq, geo)
            sel["pts"], sel["persp"] = esq, persp
            sel["datos"].update({
                "esquinas": metodo_esq, "metodo": metodo,
                "angulo": geo["ang_sup"], "inclinacion": geo["inclinacion"],
                "distorsion": geo["distorsion"],
                "ancho": int(dimension_mayor(esq)),
                "nit": medir_nitidez(persp), "ref": medir_reflejos(persp),
            })

            # ---- Guardar en capturas/imagen_N/ (una carpeta por escaneo) ----
            carpeta = nueva_carpeta_captura()
            guardar_deteccion_original(
                sel["orig"], sel["pts"], os.path.join(carpeta, "0_deteccion.jpg"), sel["datos"]
            )
            cv2.imwrite(os.path.join(carpeta, "1_original.jpg"), sel["orig"])
            cv2.imwrite(os.path.join(carpeta, "2_homografia.jpg"), sel["persp"])

            ruta_final = os.path.join(carpeta, "3_limpia.jpg")
            cv2.imwrite(ruta_final,
                        mejorar_imagen(sel["persp"], sel["luz"], final=True))

            d = sel["datos"]
            print("\n" + "=" * 50)
            print("       CAPTURA COMPLETADA")
            print("=" * 50)
            print(f"Confianza YOLO : {d['conf']:.3f}")
            print(f"Nitidez        : {d['nit']:.2f}")
            print(f"Reflejos       : {d['ref']:.2f}%")
            print(f"Iluminacion    : {d['brillo']:.2f}")
            print(f"Movimiento     : {d['mov']:.0f} px/s")
            print(f"Inclinacion    : {d['inclinacion']:.2f} grados")
            print(f"Distorsion     : {d['distorsion'] * 100:.1f}%")
            print(f"Esquinas       : {d['esquinas']}")
            print(f"Correccion     : {d['metodo']}")
            print("=" * 50)
            print(f"Guardado en {carpeta}/")
            abrir_carpeta(carpeta)
            print("  0_deteccion.jpg | 1_original.jpg | 2_homografia.jpg | 3_limpia.jpg")
            break

    cap.release()
    cv2.destroyAllWindows()
    return ruta_final

# 13. EJECUCION
if __name__ == "__main__":
    ruta_modelo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
    capturar_y_procesar(ruta_modelo)
