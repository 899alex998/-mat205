# -mat205
# Modelado Predictivo - Regresión Lineal Numérica

Este proyecto implementa de manera nativa y modular el método numérico de **Regresión Lineal por Mínimos Cuadrados** para resolver problemas del **Escenario A** (Predicción continua desde variables discretas).

## Caso de Uso Aplicado
Predicción del tiempo de latencia/respuesta en milisegundos de un servidor en función de la carga de usuarios discretos conectados simultáneamente.

## Estructura del Código
* `modelo.py`: Aloja la lógica y fórmulas del algoritmo matemático.
* `app.py`: Script ejecutable que inicializa el dataset, entrena y realiza la inferencia.

## Requisitos y Dependencias
Este desarrollo fue diseñado bajo principios de código limpio y eficiente **utilizando únicamente la librería estándar de Python 3.x**. No requiere la instalación de entornos virtuales complejos ni paquetes de terceros (`numpy` o `scikit-learn`), garantizando un despliegue inmediato.

## Instrucciones de Ejecución
1. Descargue o clone los archivos en un mismo directorio.
2. Abra una terminal en dicha ruta.
3. Ejecute el siguiente comando:
   ```bash
   python app.py
