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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Carga de Datos",
    "2. Exploración",
    "3. Limpieza",
    "4. Preparación",
    "5. Entrenamiento",
    "6. Interpretación",
    "7. Predicción"
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
                st.dataframe(info_as_dataframe(st.session_state.data), width='content')
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


# --- Pestaña 3: Preparación de Datos (Visualización Estática) ---
with tab3:
    st.header("📊 Paso 3: Proceso de Limpieza y Preparación")
    
    st.markdown("""
    ### 🎯 Pipeline Completo de Transformación
    
    Este análisis muestra el proceso exhaustivo aplicado al dataset original de *1,000,000 registros* 
    hasta obtener un dataset limpio y optimizado para modelado predictivo.
    """)
    
    # ============================================================
    # RESUMEN EJECUTIVO
    # ============================================================
    st.subheader("📈 Resumen Ejecutivo del Proceso")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Inicial", "1,000,000", help="Propiedades en Colombia")
    with col2:
        st.metric("Dataset Final", "~21,000", delta="-97.9%", help="Después de filtros y limpieza completa")
    with col3:
        st.metric("Conservación", "~2.1%", help="Datos de alta calidad preservados")
    with col4:
        st.metric("Variables Finales", "15", help="Campos relevantes")
    
    st.divider()
    
    # ============================================================
    # PASO 1: FILTRADO GEOGRÁFICO
    # ============================================================
    with st.expander("🌍 *PASO 1: Filtrado Geográfico - Antioquia*", expanded=False):
        st.markdown("### 🎯 Decisión Estratégica")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            *¿Por qué Antioquia?*
            - 🏙 Mercado inmobiliario homogéneo
            - 📊 Volumen suficiente (341,453 registros)
            - 🎯 Reduce variabilidad geográfica extrema
            - 📍 Centro económico: Medellín
            """)
        with col2:
            # Datos de distribución por departamento
            st.markdown("""
            *Top 5 Departamentos:*
            1. Antioquia: *341,453* (34.1%)
            2. Cundinamarca: 208,918 (20.9%)
            3. Valle del Cauca: 117,770 (11.8%)
            4. Atlántico: 78,605 (7.9%)
            5. Santander: 71,737 (7.2%)
            """)
        
        # Métricas del filtrado
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Antes", "1,000,000")
        with col2:
            st.metric("Después", "341,453")
        with col3:
            st.metric("Conservado", "34.1%")
        
        # Ciudades principales
        st.markdown("### 🏙 Principales Ciudades en Antioquia")
        ciudades_data = {
            'Ciudad': ['Medellín', 'Envigado', 'Sabaneta', 'Bello', 'Rionegro', 'Itagüí', 'La Estrella', 'La Ceja'],
            'Propiedades': [262856, 24171, 10836, 8728, 8166, 7566, 2206, 1921],
            'Porcentaje': ['77.0%', '7.1%', '3.2%', '2.6%', '2.4%', '2.2%', '0.6%', '0.6%']
        }
        st.dataframe(pd.DataFrame(ciudades_data), use_container_width=True, hide_index=True)
    
    # ============================================================
    # PASO 2: LIMPIEZA DE PRECIOS
    # ============================================================
    with st.expander("💰 *PASO 2: Validación de Precios*", expanded=False):
        st.markdown("### 🎯 Objetivo")
        st.info("Eliminar registros con precios inválidos y garantizar moneda homogénea")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            *Filtros Aplicados:*
            - ✅ Solo moneda COP (Pesos Colombianos)
            - ✅ Precios > 0
            - ❌ Eliminados USD, ARS, NaN
            """)
            
            st.metric("Registros Eliminados", "88", delta="-0.03%")
        
        with col2:
            st.markdown("""
            *Distribución de Monedas (Original):*
            - COP: 341,366 (99.97%)
            - nan: 79 (0.02%)
            - USD: 7 (0.00%)
            - ARS: 1 (0.00%)
            """)
            
            st.metric("Dataset Después", "341,365")
        
        st.success("✅ *Resultado:* 100% de los registros conservados tienen precios válidos en COP")
    
    # ============================================================
    # PASO 3: VALIDACIÓN GEOGRÁFICA
    # ============================================================
    with st.expander("🗺 *PASO 3: Validación de Coordenadas*", expanded=False):
        st.markdown("### 📍 Límites Geográficos de Antioquia")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            *Rangos Válidos:*
            - *Latitud:* 5.4° a 8.8°
            - *Longitud:* -77.2° a -73.8°
            
            *Antes de Validación:*
            - Latitud: -75.640 a 51.801
            - Longitud: -97.494 a 100.477
            """)
        
        with col2:
            st.markdown("""
            *Resultados:*
            - 🔍 Registros con coordenadas: 144,647
            - 🚨 Coordenadas fuera de Antioquia: *291*
            - ✅ Eliminados: 291 registros
            """)
            
            st.metric("Conservación", "99.9%")
        
        st.warning("*Justificación:* Coordenadas fuera del departamento indican errores de geolocalización")
    
    # ============================================================
    # PASO 4: FILTRADO POR TIPO DE PROPIEDAD
    # ============================================================
    with st.expander("🏠 *PASO 4: Filtrado por Tipo de Propiedad*", expanded=False):
        st.markdown("### 🎯 Enfoque Residencial")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            *Tipos Incluidos:*
            - ✅ Apartamento (236,178)
            - ✅ Casa (41,438)
            
            *Tipos Excluidos:*
            - ❌ Lote (15,344)
            - ❌ Otro (38,434)
            - ❌ Local comercial (4,681)
            - ❌ Oficina (3,623)
            - ❌ Finca (1,140)
            - ❌ Otros (848)
            """)
        
        with col2:
            st.metric("Antes", "341,074")
            st.metric("Después", "277,616")
            st.metric("Eliminados", "63,458", delta="-18.6%")
            st.metric("Conservado", "81.4%")
        
        st.info("*Justificación:* Vivienda residencial (casa/apartamento) tiene dinámicas de precio homogéneas")
    
    # ============================================================
    # PASO 5: FILTRO DE OPERACIÓN (VENTA)
    # ============================================================
    with st.expander("🏷 *PASO 5: Solo Propiedades en Venta*", expanded=False):
        st.markdown("### 🎯 Enfoque en Mercado de Venta")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            *Distribución Original:*
            - Venta: 140,435 (50.6%)
            - Arriendo: 137,127 (49.4%)
            - Arriendo temporal: 54 (0.0%)
            """)
        
        with col2:
            st.metric("Antes", "277,616")
            st.metric("Después", "140,435")
            st.metric("Eliminados", "137,181", delta="-49.4%")
        
        st.success("✅ *Resultado:* Dataset enfocado exclusivamente en propiedades en venta")
    
    # ============================================================
    # PASO 6: TEXT MINING - INNOVACIÓN CLAVE 💎
    # ============================================================
    with st.expander("⛏ *PASO 6: Text Mining - Extracción de Datos* 💎", expanded=True):
        st.markdown("### 🌟 Innovación Técnica Principal")
        
        st.info("""
        *💡 Concepto:* Muchas propiedades tienen información valiosa en el campo description 
        pero no en campos estructurados. El text mining recupera estos datos usando expresiones regulares avanzadas.
        """)
        
        # Superficie
        st.markdown("#### 📐 Extracción de Superficie")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sin Superficie", "137,798")
        with col2:
            st.metric("Extraídas", "45,755", delta="+33.2%")
        with col3:
            st.metric("Tasa Recuperación", "33.2%")
        
        # Habitaciones
        st.markdown("#### 🛏 Extracción de Habitaciones")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sin Bedrooms", "102,070")
        with col2:
            st.metric("Extraídas", "81,447", delta="+79.8%")
        with col3:
            st.metric("Tasa Recuperación", "79.8%")
        
        # Baños
        st.markdown("#### 🚿 Extracción de Baños")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sin Bathrooms", "18,727")
        with col2:
            st.metric("Extraídos", "10,795", delta="+57.6%")
        with col3:
            st.metric("Tasa Recuperación", "57.6%")
        
        st.markdown("---")
        st.success(f"""
        *🎯 TOTAL DATOS RECUPERADOS: 137,997 valores*
        
        Esta técnica de text mining es una *innovación clave* que recupera información 
        que de otra manera se perdería, mejorando significativamente la completitud del dataset.
        """)
        
        # Ejemplo de patrones
        with st.expander("🔍 Ver Patrones de Extracción Usados"):
            st.code("""
# Patrones para Superficie (m²)
- r'(\d+(?:[.,]\d+)?)\s*(?:m2|m²|metros\s*cuadrados)'
- r'(\d+(?:[.,]\d+)?)\s*(?:mts2|mt2|metros2)'
- r'(?:área|area)\s*(?:de\s*)?(\d+(?:[.,]\d+)?)'

# Patrones para Habitaciones
- r'(\d+)\s*(?:habitación|habitacion|dormitorio)(?:es)?'
- r'(\d+)\s*hab\.?(?:s)?[^a-z]'

# Patrones para Baños
- r'(\d+)\s*(?:baño|bano)(?:s)?'
- r'(\d+)\s*bath(?:s)?[^a-z]'
            """, language="python")
    
    # ============================================================
    # PASO 7: EXTRACCIÓN DE UBICACIÓN
    # ============================================================
    with st.expander("🌆 *PASO 7: Extracción de Ubicación (Ciudades y Barrios)*", expanded=False):
        st.markdown("### 📍 Text Mining de Ubicaciones")
        
        # Ciudades
        st.markdown("#### 🏙 Ciudades")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original (l3)", "137,499", help="97.9%")
        with col2:
            st.metric("Final (l3_final)", "139,853", help="99.6%")
        with col3:
            st.metric("Ganancia", "+2,354")
        
        # Barrios
        st.markdown("#### 🏘 Barrios")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original (l4)", "32,685", help="23.3%")
        with col2:
            st.metric("Final (l4_final)", "57,533", help="41.0%")
        with col3:
            st.metric("Ganancia", "+24,848")
        
        st.success("""
        ✅ *Mejora Significativa:* La extracción desde description y title 
        aumentó la cobertura de barrios de 23.3% a 41.0%
        """)
    
    # ============================================================
    # PASO 8: INTEGRACIÓN DE DATOS
    # ============================================================
    with st.expander("🔗 *PASO 8: Integración de Datos*", expanded=False):
        st.markdown("### 🎯 Estrategia de Consolidación")
        
        st.info("""
        *Prioridad:* Datos originales > Extraídos de description > Extraídos de title
        
        Se crean variables _final que combinan la mejor información disponible
        """)
        
        # Tabla de integración
        integracion_data = {
            'Variable': ['surface_total', 'bedrooms', 'bathrooms'],
            'Antes (Faltantes)': ['137,798', '102,070', '18,727'],
            'Después (Faltantes)': ['92,043', '20,623', '7,932'],
            'Valores Completados': ['45,755', '81,447', '10,795'],
            'Mejora': ['33.2%', '79.8%', '57.6%']
        }
        st.dataframe(pd.DataFrame(integracion_data), use_container_width=True, hide_index=True)
        
        st.metric("Total Valores Completados", "137,997", delta="Mejora en completitud")
    
    # ============================================================
    # PASO 9: FILTRADO FINAL - VALORES COMPLETOS
    # ============================================================
    with st.expander("✂ *PASO 9: Filtrado Final - Solo Registros Completos*", expanded=False):
        st.markdown("### 🎯 Preparación para Modelado ML")
        
        st.warning("""
        *Decisión Crítica:* Para entrenar modelos de Machine Learning efectivos, 
        se eliminan registros con valores faltantes en variables clave.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            *Variables Requeridas:*
            - ✅ surface_total_final
            - ✅ bedrooms_final
            - ✅ bathrooms_final
            - ✅ property_type
            - ✅ lat / lon
            - ✅ price
            """)
        
        with col2:
            st.metric("Antes del Filtro", "140,435")
            st.metric("Después del Filtro", "~21,000", delta="-85%")
            st.metric("Registros Eliminados", "~119,000")
        
        st.info("""
        *Justificación Técnica:*
        - Modelos ML requieren datos completos para predicciones precisas
        - Variables como superficie, habitaciones y baños son críticas para predecir precio
        - Mejor ~21K registros de alta calidad que 140K con datos faltantes
        """)
        
        st.success("✅ *Dataset Final:* Registros 100% completos en variables predictoras")
    
    # ============================================================
    # RESUMEN FINAL DEL PROCESO
    # ============================================================
    st.divider()
    st.subheader("📋 Resumen Final del Pipeline")
    
    # Tabla resumen de todos los pasos
    resumen_pipeline = {
        'Paso': [
            '1️⃣ Filtrado Geográfico',
            '2️⃣ Limpieza Precios',
            '3️⃣ Validación Coordenadas',
            '4️⃣ Filtrado Tipo Propiedad',
            '5️⃣ Solo Ventas',
            '6️⃣ Text Mining Superficie',
            '6️⃣ Text Mining Bedrooms',
            '6️⃣ Text Mining Bathrooms',
            '7️⃣ Text Mining Ubicación',
            '8️⃣ Integración Final',
            '9️⃣ Filtrado Completos'
        ],
        'Antes': [
            '1,000,000',
            '341,453',
            '341,365',
            '341,074',
            '277,616',
            '137,798 faltantes',
            '102,070 faltantes',
            '18,727 faltantes',
            '32,685 barrios',
            'Variables separadas',
            '140,435'
        ],
        'Después': [
            '341,453',
            '341,365',
            '341,074',
            '277,616',
            '140,435',
            '92,043 faltantes',
            '20,623 faltantes',
            '7,932 faltantes',
            '57,533 barrios',
            'Variables finales',
            '~21,000'
        ],
        'Impacto': [
            '-658,547',
            '-88',
            '-291',
            '-63,458',
            '-137,181',
            '+45,755 recuperados',
            '+81,447 recuperados',
            '+10,795 recuperados',
            '+24,848 recuperados',
            '137,997 datos mejorados',
            '-119,435 (sin datos completos)'
        ]
    }
    
    st.dataframe(
        pd.DataFrame(resumen_pipeline),
        use_container_width=True,
        hide_index=True
    )
    
    # ============================================================
    # ESTADÍSTICAS FINALES DEL DATASET
    # ============================================================
    st.divider()
    st.subheader("📊 Dataset Final - Estadísticas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ✅ Variables Clave")
        st.markdown("""
        - *price* (100% completo)
        - *surface_total_final* (100% completo)
        - *bedrooms_final* (100% completo)
        - *bathrooms_final* (100% completo)
        - *lat/lon* (100% completo)
        - *property_type* (100% completo)
        """)
    
    with col2:
        st.markdown("### 🏘 Distribución Geográfica")
        st.markdown("""
        - *Ciudades:* ~21,000 (100%)
        - *Barrios:* Variado (disponible)
        - *Coordenadas:* ~21,000 (100%)
        - *Departamento:* Antioquia (100%)
        """)
    
    with col3:
        st.markdown("### 📐 Completitud")
        st.markdown("""
        - Variables críticas: *100%*
        - Geolocalización: *100%*
        - Metadata: *100%*
        - *Dataset óptimo para ML*
        """)
    
    # ============================================================
    # CONCLUSIÓN
    # ============================================================
    st.divider()
    st.success("""
    ### 🎉 Pipeline de Limpieza Completado Exitosamente
    
    *Logros Principales:*
    - ✅ Reducción de 1M a ~21K registros (2.1% conservado) con criterios técnicos rigurosos
    - ✅ *137,997 datos recuperados* mediante text mining innovador
    - ✅ *100% completitud* en todas las variables clave para modelado
    - ✅ Dataset homogéneo enfocado en Antioquia (mercado inmobiliario específico)
    - ✅ Enfoque en calidad sobre cantidad: registros perfectamente completos
    - ✅ *Listo para entrenamiento de modelos de Machine Learning*
    
    *Filosofía:* Mejor ~21K registros de altísima calidad que 140K con datos faltantes
    
    *Próximo Paso:* Entrenar modelos de predicción de precios con este dataset premium
    """)
    
    # Botón informativo
    if st.button("📥 Ver Estructura del Dataset Final", type="primary"):
        st.code("""
        Dataset Final: properties_gold.csv
        
        Columnas (15):
        ├── ad_type           : Tipo de anuncio
        ├── start_date        : Fecha inicio
        ├── end_date          : Fecha fin
        ├── created_on        : Fecha creación
        ├── lat               : Latitud (100% completo)
        ├── lon               : Longitud (100% completo)
        ├── price             : Precio en COP (100% completo)
        ├── title             : Título del anuncio
        ├── description       : Descripción completa
        ├── property_type     : Tipo (Casa/Apartamento)
        ├── operation_type    : Operación (Venta)
        ├── surface_total_final   : Superficie m² (100% completo)
        ├── bedrooms_final        : Habitaciones (100% completo)
        ├── bathrooms_final       : Baños (100% completo)
        ├── l3_final              : Ciudad (100% completo)
        └── l4_final              : Barrio (disponible)
        
        Total Registros: ~21,000
        Completitud: 100% en variables críticas
        Tamaño: ~5 MB
        Calidad: Premium - Sin valores faltantes en predictores
        """, language="text")

# --- Pestaña 3: Preparación de Datos ---
with tab4:
    st.header("Paso 3: Preparar los Datos para el Modelo")
    if st.session_state.data is not None:
        if st.button("Preparar Datos"):
            with st.spinner("Dividiendo y preprocesando los datos..."):
                X_train, X_test, y_train, y_test, preprocessor = preparar_datos(st.session_state.data)
                
                # Guardamos los resultados en el estado de la sesión
                st.session_state.prepared_data = (X_train, X_test, y_train, y_test)
                st.session_state.preprocessor = preprocessor
                
                st.success("Datos preparados exitosamente.")
                st.info(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
                st.info(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")
                st.write("Vista previa de los datos de entrenamiento (X_train):")
                st.dataframe(X_train.head())
    else:
        st.warning("Por favor, carga los datos en la Pestaña 1 (Cargar Datos) primero.")

      






# --- Pestaña 3: Preparación de Datos ---
# with tab3:
#     st.header("Paso 3: Preparar los Datos para el Modelo")
#     if st.session_state.data is not None:
#         if st.button("Preparar Datos"):
#             with st.spinner("Dividiendo y preprocesando los datos..."):
#                 st.markdown(
#                     """
# **Proceso de limpieza y preparación de la data:**

# 1. **Filtro por Antioquia:** Se seleccionan solo los registros correspondientes al departamento de Antioquia.
# 2. **Limpieza de datos con coordenadas fuera de Antioquia:** Se eliminan registros con coordenadas geográficas incorrectas.
# 3. **Limpieza de valores inválidos:** Se corrigen o eliminan datos inconsistentes.
# 4. **Eliminar columnas con 0 registros (l5 y l6):** Se eliminan columnas vacías.
# 5. **Filtrar solo valores en pesos colombianos:** Se conservan solo los registros con precios en COP.
# 6. **Filtrar por tipos de propiedad:** Solo se incluyen Apartamentos y Casas.
# 7. **Eliminación de la columna Rooms:** Se elimina por ser idéntica a bedrooms.
# 8. **Filtrar solo propiedades en venta:** Se eliminan registros de arriendo o arriendo temporal.
# 9. **Recuperación de Área desde la columna descripción:** Se extrae el área en m² desde el texto.
# 10. **Recuperación de # de baños y # de bedrooms desde la columna descripción:** Se extraen estos valores desde el texto.
# 11. **Recuperación de ubicaciones como barrios y ciudades desde la columna descripción y titles:** Se extraen ubicaciones relevantes desde los textos descriptivos.
#                     """,
#                     unsafe_allow_html=True
#                 )
#                 st.subheader("dataframe preparado")
#                 df_preparado = preparar_datos()

                
#                 st.dataframe(df_preparado.head())
#                 st.subheader("Cantidad de registros")
#                 # Mostrar cantidad de registros
#                 st.write(
#                     f"Registros: {df_preparado.shape[0]}, "
#                     f"Columnas: {df_preparado.shape[1]}"
#                 )
#                 st.dataframe(info_as_dataframe(df_preparado))
#                 st.session_state.data = df_preparado



#                 # X_train, X_test, y_train, y_test, preprocessor = preparar_datos(st.session_state.data)
                
#                 # # Guardamos los resultados en el estado de la sesión
#                 # st.session_state.prepared_data = (X_train, X_test, y_train, y_test)
#                 # st.session_state.preprocessor = preprocessor
                
#                 # st.success("Datos preparados exitosamente.")
#                 # st.info(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
#                 # st.info(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")
#                 # st.write("Vista previa de los datos de entrenamiento (X_train):")
#                 # st.dataframe(X_train.head())
#     else:
#         st.warning("Por favor, carga los datos en la Pestaña 1 (Cargar Datos) primero.")

