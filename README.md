# Habidata Project

## Descripción
Habidata Project es una aplicación diseñada para analizar y predecir precios de propiedades en Colombia. Utiliza técnicas de Machine Learning y herramientas de visualización para proporcionar insights sobre datos inmobiliarios.

## Características
- **Análisis Exploratorio de Datos (EDA):** Visualización de distribuciones, correlaciones y detección de valores atípicos.
- **Modelos de Machine Learning:** Entrenamiento y evaluación de modelos como Random Forest, Gradient Boosting, y más.
- **Predicción de Precios:** Predicción de precios de propiedades basándose en características como superficie, número de habitaciones, baños, y ubicación.
- **Interfaz de Usuario:** Aplicación interactiva desarrollada con Streamlit para facilitar la interacción con los datos y modelos.

## Requisitos
Asegúrate de tener instaladas las siguientes dependencias antes de ejecutar el proyecto. Puedes instalar las librerías necesarias con el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto
```
habidata-project/
│
├── app-front/               # Contiene la aplicación Streamlit
│   ├── app.py              # Archivo principal de la aplicación
│   ├── main.py             # Funciones auxiliares para la aplicación
│
├── data/                   # Datos utilizados para el análisis
│
├── etl-modules/            # Notebooks y scripts para limpieza y transformación de datos
│
├── requirements.txt        # Dependencias del proyecto
│
└── README.md               # Documentación del proyecto
```

## Cómo Ejecutar el Proyecto
1. Clona este repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   ```

2. Navega al directorio del proyecto:
   ```bash
   cd habidata-project
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Ejecuta la aplicación Streamlit:
   ```bash
   streamlit run app-front/app.py
   ```

## Tecnologías Utilizadas
- **Lenguaje:** Python
- **Frameworks:** Streamlit, Scikit-learn
- **Visualización:** Matplotlib, Seaborn
- **Base de Datos:** SQLAlchemy, Psycopg2

## Contribuciones
Las contribuciones son bienvenidas. Si deseas contribuir, por favor sigue estos pasos:
1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir nueva funcionalidad'`).
4. Envía un pull request.

## Licencia
Este proyecto está bajo la Licencia MIT. Para más detalles, consulta el archivo LICENSE.