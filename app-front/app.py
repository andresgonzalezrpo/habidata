import sys
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns


try:
    from main import (
       cargar_datos,
       explorar_datos,
       preparar_datos
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

# --- Pestaña 2: Análisis Exploratorio de Datos (EDA) ---
def info_as_dataframe(df):
    info = {
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum(),
        "Dtype": df.dtypes.astype(str)
    }
    return pd.DataFrame(info)

with tab2:
    st.header("Paso 2: Análisis Exploratorio de Datos (EDA)")
    if st.session_state.data is not None:
        if st.button("Explorar Datos"):
            with st.spinner("Generando visualizaciones..."):
                
                df_explorado = explorar_datos(st.session_state.data.copy())
                st.subheader("Cantidad de registros")
                # Mostrar cantidad de registros
                st.write(
                    f"Registros: {df_explorado['Cantidad de registros'][0]}, "
                    f"Columnas: {df_explorado['Cantidad de registros'][1]}"
                )

                st.subheader("Info")
                st.dataframe(info_as_dataframe(st.session_state.data))
                st.subheader("Estadísticas descriptivas")
                st.dataframe(df_explorado["Estadísticas descriptivas"])
                st.subheader("Valores faltantes por columna")
                missing = st.session_state.data.isnull().sum().reset_index()
                missing.columns = ['Columna', 'Valores Faltantes']
                missing['% Faltantes'] = (missing['Valores Faltantes'] / len(st.session_state.data) * 100).round(2)
                st.dataframe(missing, width='content')
                # columnas categoricas y numericas
                st.subheader("Tipos de columnas")
                st.markdown("**Columnas numéricas:**<br>" + ", ".join(df_explorado['Columnas numéricas']), unsafe_allow_html=True)
                st.markdown("**Columnas categóricas:**<br>" + ", ".join(df_explorado['Columnas categóricas']), unsafe_allow_html=True)
                # Distribución de variables númericas
                st.subheader("Distribución de variables numéricas")
                st.image("boxplots_numericas.png", caption="Distribución de variables numéricas")
                # correlaciones entre variables numéricas
                st.subheader("Matriz de correlación entre variables numéricas")
                st.image('matriz_correlacion.png', caption='Matriz de correlación')                
               
                st.session_state.data = df_explorado
                st.success("Análisis exploratorio completado.")
    else:
        st.warning("Por favor, carga los datos en la Pestaña 1 (Cargar Datos) primero.")



# --- Pestaña 3: Preparación de Datos ---
with tab3:
    st.header("Paso 3: Preparar los Datos para el Modelo")
    if st.session_state.data is not None:
        if st.button("Preparar Datos"):
            with st.spinner("Dividiendo y preprocesando los datos..."):
                st.markdown(
                    """
**Proceso de limpieza y preparación de la data:**

1. **Filtro por Antioquia:** Se seleccionan solo los registros correspondientes al departamento de Antioquia.
2. **Limpieza de datos con coordenadas fuera de Antioquia:** Se eliminan registros con coordenadas geográficas incorrectas.
3. **Limpieza de valores inválidos:** Se corrigen o eliminan datos inconsistentes.
4. **Eliminar columnas con 0 registros (l5 y l6):** Se eliminan columnas vacías.
5. **Filtrar solo valores en pesos colombianos:** Se conservan solo los registros con precios en COP.
6. **Filtrar por tipos de propiedad:** Solo se incluyen Apartamentos y Casas.
7. **Eliminación de la columna Rooms:** Se elimina por ser idéntica a bedrooms.
8. **Filtrar solo propiedades en venta:** Se eliminan registros de arriendo o arriendo temporal.
9. **Recuperación de Área desde la columna descripción:** Se extrae el área en m² desde el texto.
10. **Recuperación de # de baños y # de bedrooms desde la columna descripción:** Se extraen estos valores desde el texto.
11. **Recuperación de ubicaciones como barrios y ciudades desde la columna descripción y titles:** Se extraen ubicaciones relevantes desde los textos descriptivos.
                    """,
                    unsafe_allow_html=True
                )
                st.subheader("dataframe preparado")
                df_preparado = preparar_datos()

                
                st.dataframe(df_preparado.head())
                st.subheader("Cantidad de registros")
                # Mostrar cantidad de registros
                st.write(
                    f"Registros: {df_preparado.shape[0]}, "
                    f"Columnas: {df_preparado.shape[1]}"
                )
                st.dataframe(info_as_dataframe(df_preparado))
                st.session_state.data = df_preparado



                # X_train, X_test, y_train, y_test, preprocessor = preparar_datos(st.session_state.data)
                
                # # Guardamos los resultados en el estado de la sesión
                # st.session_state.prepared_data = (X_train, X_test, y_train, y_test)
                # st.session_state.preprocessor = preprocessor
                
                # st.success("Datos preparados exitosamente.")
                # st.info(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
                # st.info(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")
                # st.write("Vista previa de los datos de entrenamiento (X_train):")
                # st.dataframe(X_train.head())
    else:
        st.warning("Por favor, carga los datos en la Pestaña 1 (Cargar Datos) primero.")

