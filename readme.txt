================================================================================
   SISTEMA INTELIGENTE DE ESTIMACION DE DEMANDA DE AGROQUIMICOS
   Instrucciones de Instalacion, Ejecucion y Verificacion de la Demo
================================================================================

1. PRERREQUISITOS DE SOFTWARE
--------------------------------------------------------------------------------
- Python 3.10 o superior instalado.
- Gestor de paquetes pip actualizado.
- Git instalado y configurado en la terminal.
- Navegador web moderno (Google Chrome, Microsoft Edge o Firefox).


2. PASO A PASO DE INSTALACION
--------------------------------------------------------------------------------
Paso 1: Clonar el repositorio y acceder a la carpeta:
    git clone https://github.com/carlosAAF25/ml-demand-forecasting.git
    cd ml-demand-forecasting

Paso 2: Crear el entorno virtual de Python:
    python -m venv venv

Paso 3: Activar el entorno virtual:
    - En Windows (Git Bash):
        source venv/Scripts/activate
    - En Windows (PowerShell):
        .\venv\Scripts\Activate.ps1
    - En Linux / macOS:
        source venv/bin/activate

Paso 4: Instalar las dependencias fijadas del proyecto:
    pip install --upgrade pip
    pip install -r requirements.txt


3. REPRODUCIBILIDAD Y ENTRENAMIENTO (OPCIONAL)
--------------------------------------------------------------------------------
Si deseas regenerar el dataset sintetico determinista (1800 muestras, semilla 42)
y reentrenar los modelos con impresion de metricas en consola:

    python generate_and_train.py

Esto generara el artefacto final: "demand_forecaster.joblib".


4. EJECUCION DE LA DEMO (BACKEND Y FRONTEND)
--------------------------------------------------------------------------------
Paso 1: Iniciar el servidor API de FastAPI:
    uvicorn backend.app:app --reload --port 8000

    * La API estara escuchando en: http://127.0.0.1:8000
    * La documentacion interactiva (Swagger): http://127.0.0.1:8000/docs

Paso 2: Abrir la interfaz de usuario:
    - Abre directamente con doble clic el archivo "frontend/index.html" en tu navegador.
    - O ejecutalo mediante Live Server en VS Code si lo tienes instalado.


5. EJECUCION DE PRUEBAS AUTOMATIZADAS
--------------------------------------------------------------------------------
Para verificar la suite de pruebas unitarias y de integracion de la API:

    pytest tests/test_api.py -v

Casos de prueba cubiertos:
1. test_predict_valid_payload: Inferencia nominal con codigos 200 OK.
2. test_predict_edge_case_off_season: Caso borde fuera de temporada / baja demanda.
3. test_predict_invalid_week_rejected: Validacion de semana invalida (HTTP 422).
================================================================================
