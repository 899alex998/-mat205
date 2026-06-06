# -mat205
# ============================================================
#  Regresion Logistica Binaria por Gradiente Descendiente
#  Escenario C: Variables Continuas -> Variable Discreta
#  Caso de uso: Prediccion de fallos en servidores TI
#  MAT205-LAB | Ingenieria de Sistemas
# ============================================================

import numpy as np


# ------------------------------------------------------------
# 1. FUNCION SIGMOIDE
# ------------------------------------------------------------

def sigmoide(z):
    """
    Transforma cualquier valor real en una probabilidad [0, 1].
    Se aplica clipping para evitar desbordamiento numerico (overflow).
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------
# 2. FUNCION DE COSTO — Entropia Cruzada Binaria
# ------------------------------------------------------------

def calcular_costo(yhat, y):
    """
    Mide la discrepancia entre las probabilidades predichas y las etiquetas reales.
    Epsilon (1e-15) previene log(0) sin alterar el resultado para predicciones normales.
    La funcion es convexa, lo que garantiza convergencia al minimo global.
    """
    n = len(y)
    eps = 1e-15
    return -np.mean(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps))


# ------------------------------------------------------------
# 3. NORMALIZACION MIN-MAX
# ------------------------------------------------------------

def normalizar(X):
    """
    Lleva todas las variables al rango [0, 1].
    Sin normalizacion, Network_Traffic (cientos de Mbps) dominaria el gradiente,
    dejando los pesos de CPU y RAM sin actualizar durante muchas iteraciones.
    Retorna X_norm, minimos y rangos (necesarios para normalizar datos nuevos).
    """
    mins = X.min(axis=0)
    rangos = X.max(axis=0) - X.min(axis=0)
    rangos[rangos == 0] = 1  # evitar division por cero
    return (X - mins) / rangos, mins, rangos


# ------------------------------------------------------------
# 4. ENTRENAMIENTO — Gradiente Descendiente
# ------------------------------------------------------------

def entrenar(X, y, alfa=0.1, max_iter=3000, tol=1e-6):
    """
    Ajusta los pesos w y el sesgo b minimizando la Entropia Cruzada Binaria.

    Parametros:
        X        : matriz de caracteristicas normalizadas (n_muestras x n_features)
        y        : vector de etiquetas binarias (0 = Estable, 1 = Critico)
        alfa     : tasa de aprendizaje (learning rate)
        max_iter : numero maximo de iteraciones (epocas)
        tol      : tolerancia para criterio de parada por convergencia

    Retorna:
        w, b     : parametros optimizados
        historial: lista de costos por epoca (para analisis de convergencia)
        epoca    : numero de epocas hasta convergencia
    """
    n, m = X.shape
    w = np.zeros(m)
    b = 0.0
    historial = []
    costo_prev = float('inf')

    for epoca in range(max_iter):
        # Prediccion
        z = X @ w + b
        yhat = sigmoide(z)

        # Costo actual
        costo = calcular_costo(yhat, y)
        historial.append(costo)

        # Gradientes
        error = yhat - y
        grad_w = (X.T @ error) / n
        grad_b = np.mean(error)

        # Actualizacion de parametros
        w -= alfa * grad_w
        b -= alfa * grad_b

        # Criterio de parada por convergencia
        if abs(costo_prev - costo) < tol:
            print(f"Convergencia en epoca {epoca + 1}")
            return w, b, historial, epoca + 1

        costo_prev = costo

    print(f"Se alcanzo el maximo de {max_iter} iteraciones")
    return w, b, historial, max_iter


# ------------------------------------------------------------
# 5. PREDICCION
# ------------------------------------------------------------

def predecir(X_norm, w, b, umbral=0.5):
    """
    Calcula la probabilidad de colapso y la clase predicha para cada servidor.

    Parametros:
        X_norm  : datos de entrada ya normalizados
        w, b    : parametros del modelo entrenado
        umbral  : punto de corte para clasificacion (default 0.5 = 50%)

    Retorna:
        probabilidades : vector de probabilidades de colapso [0.0 - 1.0]
        clases         : vector de etiquetas predichas (0 o 1)
    """
    probabilidades = sigmoide(X_norm @ w + b)
    clases = (probabilidades >= umbral).astype(int)
    return probabilidades, clases


# ------------------------------------------------------------
# 6. EVALUACION DEL MODELO
# ------------------------------------------------------------

def evaluar(y_real, y_pred):
    """
    Calcula metricas de clasificacion: exactitud, precision, recall y F1-score.
    """
    tp = np.sum((y_pred == 1) & (y_real == 1))
    tn = np.sum((y_pred == 0) & (y_real == 0))
    fp = np.sum((y_pred == 1) & (y_real == 0))
    fn = np.sum((y_pred == 0) & (y_real == 1))

    exactitud  = (tp + tn) / len(y_real)
    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"exactitud": exactitud, "precision": precision, "recall": recall, "f1": f1}


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    # ----------------------------------------------------------
    # DATASET: Telemetria de servidores TI
    # Columnas: CPU_Usage(%), RAM_Usage(%), Network_Traffic(Mbps)
    # Etiqueta:  0 = Servidor Estable | 1 = Servidor Critico
    # ----------------------------------------------------------
    X = np.array([
        [25,  30,  120],
        [40,  45,  180],
        [55,  60,  250],
        [35,  38,  160],
        [30,  35,  140],
        [45,  50,  200],
        [70,  75,  320],
        [80,  82,  380],
        [88,  90,  430],
        [92,  95,  500],
        [85,  88,  460],
        [75,  78,  350],
    ], dtype=float)

    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=float)

    # ----------------------------------------------------------
    # PREPROCESAMIENTO
    # ----------------------------------------------------------
    X_norm, mins, rangos = normalizar(X)

    # ----------------------------------------------------------
    # ENTRENAMIENTO
    # ----------------------------------------------------------
    print("=" * 52)
    print("  REGRESION LOGISTICA BINARIA — MAT205-LAB")
    print("=" * 52)
    print()

    w, b, historial, epocas = entrenar(X_norm, y, alfa=0.1, max_iter=3000, tol=1e-6)

    print()
    print("--- Parametros Optimizados ---")
    print(f"Pesos (w): {np.round(w, 4)}")
    print(f"  w_CPU     = {w[0]:.4f}")
    print(f"  w_RAM     = {w[1]:.4f}")
    print(f"  w_Trafico = {w[2]:.4f}")
    print(f"Sesgo  (b): {b:.4f}")
    print(f"Costo final: {historial[-1]:.6f}")

    # ----------------------------------------------------------
    # EVALUACION SOBRE DATOS DE ENTRENAMIENTO
    # ----------------------------------------------------------
    probs_train, preds_train = predecir(X_norm, w, b)
    metricas = evaluar(y, preds_train)

    print()
    print("--- Evaluacion sobre datos de entrenamiento ---")
    print(f"Exactitud : {metricas['exactitud']*100:.1f}%")
    print(f"Precision : {metricas['precision']*100:.1f}%")
    print(f"Recall    : {metricas['recall']*100:.1f}%")
    print(f"F1-Score  : {metricas['f1']*100:.1f}%")

    # ----------------------------------------------------------
    # EVALUACION SOBRE CASOS DE PRUEBA
    # ----------------------------------------------------------
    casos_prueba = np.array([
        [35,  40,  150],   # Caso estable claro
        [60,  65,  280],   # Zona intermedia
        [78,  80,  380],   # Caso critico medio
        [90,  95,  500],   # Caso critico extremo
    ], dtype=float)

    etiquetas_reales = np.array([0, 0, 1, 1])

    casos_norm = (casos_prueba - mins) / rangos
    probs_prueba, preds_prueba = predecir(casos_norm, w, b)

    print()
    print("--- Evaluacion sobre Casos de Prueba ---")
    print(f"{'Servidor':<10} {'CPU':>5} {'RAM':>5} {'Traf':>6}  {'Clase Real':>10}  {'P(colapso)':>12}  {'Accion':>25}")
    print("-" * 80)

    acciones = {0: "Servidor Estable", 1: "DISPARAR ALERTA"}
    for i, (caso, prob, pred, real) in enumerate(zip(casos_prueba, probs_prueba, preds_prueba, etiquetas_reales)):
        accion = acciones[pred]
        marca = "OK" if pred == real else "ERROR"
        print(f"Servidor {i+1:<2} {int(caso[0]):>4}% {int(caso[1]):>4}% {int(caso[2]):>5}Mbps  "
              f"{'Critico' if real == 1 else 'Estable':>10}  "
              f"{prob*100:>10.1f}%  "
              f"{accion:>25}  [{marca}]")

    print()
    print("=" * 52)
    print("  Fin de ejecucion — modelo_logistico.py")
    print("=" * 52)
