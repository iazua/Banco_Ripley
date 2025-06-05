# Optimización de Dotación y Predicción de Efectividad

Este proyecto se centra en la optimización de la dotación de personal y la predicción de la efectividad operativa para diferentes sucursales. Utiliza modelos de Machine Learning para predecir métricas clave y, a partir de estas predicciones, estimar la dotación óptima necesaria para alcanzar una efectividad deseada.

## Estructura del Proyecto

El proyecto está organizado en los siguientes archivos y directorios:

-   `app.py`: Contiene la lógica principal de la aplicación, incluyendo la carga de datos, la generación de predicciones, el cálculo de la dotación óptima y, potencialmente, la visualización de resultados.
-   `train_models.py`: Script encargado de entrenar los modelos predictivos para diferentes métricas (`T_AO`, `T_AO_VENTA`, `T_VISITAS`, `DOTACION`) por sucursal.
-   `preprocessing.py`: Módulo con funciones para el preprocesamiento de datos, como la creación de características temporales, rezagos y promedios móviles, y la identificación de días festivos.
-   `utils.py`: Contiene funciones auxiliares para el cálculo de efectividad, la estimación de parámetros del modelo de efectividad y la determinación de la dotación óptima.
-   `models/`: Directorio donde se guardan los modelos entrenados.
-   `data/DOTACION_EFECTIVIDAD.xlsx`: Archivo de datos de entrada con información histórica de dotación, efectividad y otras métricas.
-   `requirements.txt`: Lista de las dependencias de Python del proyecto.

## Funcionalidades Principales

1.  **Carga y Preprocesamiento de Datos**: Lee datos históricos de un archivo Excel, limpia y prepara las características para el entrenamiento de modelos.
2.  **Entrenamiento de Modelos Predictivos**: Entrena modelos XGBoost para predecir:
    *   `T_AO`: Tiempo de Atención de Operaciones
    *   `T_AO_VENTA`: Tiempo de Atención de Operaciones de Venta
    *   `T_VISITAS`: Tiempo de Visitas
    *   `DOTACION`: Dotación de personal
    Los modelos se entrenan de forma individual para cada sucursal.
3.  **Generación de Predicciones Futuras**: Utiliza los modelos entrenados para predecir las métricas mencionadas para un período futuro (ej. los próximos 7 días).
4.  **Estimación de Dotación Óptima**: Basándose en las predicciones y un modelo de efectividad sigmoide, el sistema calcula la dotación óptima necesaria para cada sucursal para lograr una efectividad deseada.

## Cómo Ejecutar el Proyecto

### 1. Configuración del Entorno

Asegúrate de tener un entorno Python (`virtualenv` es el recomendado) configurado.

```bash
python -m venv venv
./venv/Scripts/activate # En Windows
source venv/bin/activate # En Linux/macOS
```


### 2. Instalación de Dependencias

Instala las librerías necesarias utilizando `pip` y el archivo `requirements.txt`:


### 3. Preparación de Datos

Coloca tu archivo de datos históricos (`DOTACION_EFECTIVIDAD.xlsx`) en el directorio `data/`. Asegúrate de que el archivo tenga las columnas esperadas: `COD_SUC`, `FECHA`, `T_AO`, `T_AO_VENTA`, `DOTACION`, `T_VISITAS`, `P_EFECTIVIDAD`.

### 4. Entrenamiento de Modelos

Ejecuta el script de entrenamiento para generar los modelos predictivos:


Esto creará los archivos de modelo (`.pkl`) en el directorio `models/`.

### 5. Ejecución de la Aplicación

Una vez que los modelos estén entrenados, puedes ejecutar la aplicación principal:

