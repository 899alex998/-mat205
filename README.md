# -mat205
# Regresión Logística Binaria — Escenario C

## Descripción

Implementación desde cero de un modelo de **Regresión Logística Binaria** con NumPy como única dependencia, sin uso de librerías de Machine Learning de alto nivel como Scikit-Learn.

El modelo clasifica el estado de un servidor web como **Estable (clase 0)** o **Crítico/en riesgo de colapso (clase 1)** a partir de tres variables continuas de telemetría: porcentaje de uso de CPU, porcentaje de uso de RAM y volumen de tráfico de red en Mbps.

Este trabajo corresponde al **Escenario C** del Laboratorio 1: predicción de una variable discreta/categórica a partir de variables continuas, dentro del marco de Métodos Numéricos aplicados a Ingeniería de Sistemas.

---

## Estructura del repositorio

```
/
├── modelo_logistico.py   # Clase principal RegresionLogisticaNumerica + script de prueba
├── requirements.txt      # Lista de dependencias (NumPy)
└── README.md             # Este archivo
```

---

## Requisitos

- **Python 3.8** o superior
- **NumPy 1.21** o superior

Para verificar tu versión de Python:

```bash
python --version
```

---

## Instalación de dependencias

```bash
pip install numpy
```

O si usás el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Cómo ejecutar

1. Cloná o descargá este repositorio:

```bash
git clone https://github.com/TU_USUARIO/lab1-regresion-logistica-mat205.git
cd lab1-regresion-logistica-mat205
```

2. Ejecutá el script principal:

```bash
python modelo_logistico.py
```

El script ejecuta automáticamente: generación del dataset de telemetría simulado, normalización Min-Max de todas las variables, entrenamiento con Gradiente Descendiente, impresión de los parámetros optimizados y evaluación sobre casos de prueba.

---

## Salida esperada

```
Convergencia en época 312

--- Parámetros Optimizados ---
Pesos (w): [5.842  6.103  4.917]
Sesgo  (b): -5.2341
Costo final: 0.031842

--- Evaluación de Alerta Temprana ---
CPU: 90%  RAM: 95%  Tráfico: 500 Mbps
Probabilidad de colapso: 99.87%
Acción: DISPARAR AUTO-ESCALADO
```

---

## Dataset utilizado

Conjunto de datos simulado con registros históricos de telemetría de servidores:

| CPU_Usage (%) | RAM_Usage (%) | Network_Traffic (Mbps) | Clase |
|:---:|:---:|:---:|:---:|
| 25 | 30 | 120 | 0 (Estable) |
| 40 | 45 | 180 | 0 (Estable) |
| 55 | 60 | 250 | 0 (Estable) |
| 70 | 75 | 320 | 1 (Crítico) |
| 80 | 82 | 380 | 1 (Crítico) |
| 88 | 90 | 430 | 1 (Crítico) |
| 92 | 95 | 500 | 1 (Crítico) |
| 35 | 38 | 160 | 0 (Estable) |

---

## Sustento matemático

El modelo aplica la **función sigmoide** sobre una combinación lineal de las entradas para producir una probabilidad entre 0 y 1:

```
ŷ = σ(z) = 1 / (1 + e⁻ᶻ)     donde     z = wᵀx + b
```

Los parámetros `w` y `b` se optimizan minimizando la **Entropía Cruzada Binaria** mediante Gradiente Descendiente:

```
w := w − α · (1/n) · Xᵀ(ŷ − y)
b := b − α · (1/n) · Σ(ŷ − y)
```

La función de costo es convexa, lo que garantiza convergencia siempre al mínimo global.

---

## Caso de uso

**Predicción de fallos en servidores de infraestructura TI.**  
El modelo permite anticipar el colapso de un servidor antes de que ocurra, disparando alertas de auto-escalado cuando la probabilidad supera el umbral del 50%, pasando de un sistema reactivo a uno predictivo.

