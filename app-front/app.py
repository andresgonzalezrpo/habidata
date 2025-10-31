import sys
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns


try:
    from main import (
       cargar_datos,
       explorar_datos
    )
except ImportError:
    st.error("Error: no se cargaron los datos correctamente")
    st.stop()

#--- Configuración de la página ---
st.set_page_config(
    page_title="Análisis de precios para propiedades en Antioquia según sus características",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Análisis de precios para propiedades en Antioquia según sus características")
st.write("Esta aplicación interactiva te permite ver todo el proceso de análisis de datos, desde la carga y limpieza de los datos hasta la visualización y modelado predictivo" \
" utilizando un conjunto de datos de propiedades en Antioquia.")

# --- Inicialización del estado de la sesión ---
# El estado de la sesión se usa para guardar variables entre interacciones
if 'data' not in st.session_state:
    st.session_state.data = None
if 'prepared_data' not in st.session_state:
    st.session_state.prepared_data = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

# --- Creación de Pestañas ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Carga de Datos",
    "2. Exploración",
    "3. Preparación",
    "4. Entrenamiento",
    "5. Interpretación",
    "6. Predicción"
])

# --- Pestaña 1: Carga de Datos ---
with tab1:
    st.header("Paso 1: Cargar el Conjunto de Datos")
    st.info("Haz clic en el botón para cargar los datos de propiedades en Antioquia.")

    if st.button("Cargar Datos"):
        with st.spinner("Cargando datos..."):
            st.session_state.data = cargar_datos()
        st.success("¡Datos cargados exitosamente!")
        st.dataframe(st.session_state.data.head())

with tab2:
    st.header("Paso 2: Análisis Exploratorio de Datos (EDA)")
    if st.session_state.data is not None:
        if st.button("Explorar Datos"):
            with st.spinner("Generando visualizaciones..."):
                # Capturamos los prints para mostrarlos en la app
                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()

                # Ejecutamos la función y guardamos las figuras
                df_explorado = explorar_datos(st.session_state.data.copy())
                
                # Restauramos la salida estándar
                sys.stdout = old_stdout
                
                st.subheader("Información y Estadísticas")
                st.text(captured_output.getvalue())

                # st.subheader("Visualizaciones")
                # # Mostramos las imágenes guardadas por la función
                # st.image('titanic_supervivencia.png', caption='Distribución de Supervivencia')
                # st.image('titanic_supervivencia_sexo.png', caption='Tasa de Supervivencia por Sexo')
                # st.image('titanic_supervivencia_clase.png', caption='Tasa de Supervivencia por Clase')
                # st.image('titanic_edad_supervivencia.png', caption='Distribución de Edades por Supervivencia')
                # st.image('titanic_familia_supervivencia.png', caption='Tasa de Supervivencia por Tamaño de Familia')
                
                # Guardamos el dataframe con la nueva columna 'FamilySize'
                st.session_state.data = df_explorado
                st.success("Análisis exploratorio completado.")
    else:
        st.warning("Por favor, carga los datos en la Pestaña 1 (Cargar Datos) primero.")

