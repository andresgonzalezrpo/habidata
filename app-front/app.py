import sys
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os


try:
    from main import (
       cargar_datos,
       explorar_datos,
       preparar_datos,
       preparar_modelo,
       entrenar_evaluar_modelos,
       predecir_precio_vivienda
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
" utilizando un conjunto de datos de propiedades en Antioquia con precios del 2020 y 2021")

# --- Inicialización del estado de la sesión ---
# El estado de la sesión se usa para guardar variables entre interacciones
if 'data' not in st.session_state:
    st.session_state.data = None
if 'prepared_data' not in st.session_state:
    st.session_state.prepared_data = None
if 'prepared_model' not in st.session_state:
    st.session_state.prepared_model = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

# --- Creación de Pestañas ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "1. Presentación del equipo",
    "2. Carga de Datos",
    "3. Exploración",
    "4. Limpieza",
    "5. Preparación",
    "6. Entrenamiento",
    "7. Predicción",
    "8. Interpretación",
    
])

# --- Presentación del equipo ---
with tab1:
    st.header("👥 Equipo de Trabajo")
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots/miembros')
    


    # Información de los miembros del equipo
    miembros = [
        {"nombre": "Giovanny Casas Agudelo", "foto": "Gio.jpg"},
        {"nombre": "Carmen Carvajal Gutiérrez", "foto": "Carmen.jpg"},
        {"nombre": "Camilo Arango Yepes", "foto": "camilo.jpg"},
        {"nombre": "Andrés González Restrepo", "foto": "Andres1.jpg"}
    ]

    # Mostrar los miembros en columnas
    cols = st.columns(len(miembros))

    for col, miembro in zip(cols, miembros):
        with col:
            st.image(os.path.join(plots_dir, miembro["foto"]), width=150, caption=miembro["nombre"])

# --- Pestaña 1: Carga de Datos ---
with tab2:
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

with tab3:
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

                st.subheader("Descripción de las Columnas y Datos")
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
                plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')
                st.image(os.path.join(plots_dir, "boxplots_numericas.png"), caption="Distribución de variables numéricas")
                # correlaciones entre variables numéricas
                st.subheader("Matriz de correlación entre variables numéricas")
                st.image(os.path.join(plots_dir, 'matriz_correlacion.png'), caption='Matriz de correlación')    
                # histograma de la variable objetivo
                st.subheader("Histograma de barrios")
                st.image(os.path.join(plots_dir, "grafico_barrios_l4.png"), caption="Cantidad de datos por barrio")
               

                
                st.session_state.data = df_explorado
                st.success("Análisis exploratorio completado.")
    else:
        st.warning("Por favor, carga los datos en la Pestaña 1 (Cargar Datos) primero.")
# --- Pestaña 3: Preparación de Datos ---


# --- Pestaña 3: Preparación de Datos (Visualización Estática) ---
with tab4:
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
        st.dataframe(pd.DataFrame(ciudades_data), width='stretch', hide_index=True)
    
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
        st.dataframe(pd.DataFrame(integracion_data), width='stretch', hide_index=True)
        
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
        width='stretch',
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

# --- Pestaña 4: Preparación de Datos ---
with tab5:
    st.header("Paso 4: Preparar los Datos para el Modelo")   
    
    with st.spinner("Cargando datos..."):
        st.session_state.prepared_data = preparar_datos()
        data = st.session_state.prepared_data
    #st.success("¡Datos cargados exitosamente!")
    #st.dataframe(st.session_state.prepared_data.head())


    if st.session_state.prepared_data is not None:
        if st.button("Preparar Modelo"):
            # graficar boxplot de la variable objetivo sin outliers
            plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')
            st.subheader("Limpieza de outliers en la variable objetivo (precio)")
            st.image(os.path.join(plots_dir, "boxplot_precios_sin_outliers.png"), caption="Boxplot de precios de propiedades")

            st.subheader("Limpieza de outliers en el área total")
            st.image(os.path.join(plots_dir, "dispersión_surface_total_final.png"), caption="Dispersión de superficie total final")
            X_train, X_test, y_train, y_test, preprocessor = preparar_modelo(st.session_state.prepared_data)
            print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
            print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

                
            # Guardamos los resultados en el estado de la sesión
            st.session_state.prepared_model = (X_train, X_test, y_train, y_test)
            st.session_state.preprocessor = preprocessor
            
            st.success("Datos preparados exitosamente.")
            st.info(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
            st.info(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")
            st.write("Vista previa de los datos de entrenamiento (X_train):")
            st.dataframe(X_train.head())

with tab6:
    st.header("Paso 5: Entrenar y Evaluar Múltiples Modelos")
    if st.session_state.prepared_model is not None:
        if st.button("Entrenar y Evaluar Modelos"):
            with st.spinner("Entrenando modelos y evaluando... Esto puede tardar un momento."):
                X_train, X_test, y_train, y_test = st.session_state.prepared_model
                preprocessor = st.session_state.preprocessor               

                best_model = entrenar_evaluar_modelos(X_train, X_test, y_train, y_test, preprocessor)
                # Resultados de los modelos
                st.subheader("Resultados de Validación Cruzada")

                # Crear DataFrame con los resultados
                resultados = {
                    "Modelo": [
                        "Random Forest",
                        "Decision Tree",
                        "Gradient Boosting",
                        "Linear Regression",
                        "Support Vector Regressor"
                    ],
                    "R2_mean": [
                        0.843147,
                        0.795640,
                        0.781758,
                        0.701599,
                        -0.058193
                    ],
                    "MAE_mean": [
                        43344700,
                        44768540,
                        62318070,
                        70768440,
                        140069600
                    ],
                    "RMSE_mean": [
                        72722650,
                        83012630,
                        85812780,
                        100312600,
                        189028200
                    ]
                }

                df_resultados = pd.DataFrame(resultados)

                # Mostrar tabla con estilo
                st.dataframe(
                    df_resultados.style.format({
                        "R2_mean": "{:.6f}",
                        "MAE_mean": "{:,}",
                        "RMSE_mean": "{:,}"
                    }).highlight_max(axis=0, subset=["R2_mean"], color="lightgreen")
                )

                plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')
                st.image(os.path.join(plots_dir, "r2_mean_plot.png"), caption="Error del promedio r2")

                st.image(os.path.join(plots_dir, "mae_rmse_mean_plot.png"), caption="Error del promedio MAE Y RMSE")

    
                # Guardamos el mejor modelo
                st.session_state.best_model = best_model
                st.session_state.model_name = "Random Forest Regressor"
                st.success(f"Entrenamiento completado. El mejor modelo es: **{st.session_state.model_name}**")

                # Datos para la gráfica
                hiperparametros = {
                    'Métrica': ['Mejor score (R2)', 'R2 (test)', 'MAE (test)', 'RMSE (test)'],
                    'Valor': [0.8434188833882883, 0.8436063284766978, 42487513.483914204, 5253572023866604.0]
                }

                df_hiperparametros = pd.DataFrame(hiperparametros)
                st.subheader("Modelo mejorado con hipermarámetros")
                st.dataframe(df_hiperparametros)

                
    else:
        st.warning("Por favor, prepara los datos en la Pestaña 4 (Preparar el modelo) primero.")


with tab7:
    st.header("Paso 6: Simular y Realizar una Predicción de Precio")

    # Verificamos que el modelo esté cargado correctamente en la sesión
    if st.session_state.best_model is not None:
        st.info("Completa el siguiente formulario para ingresar los datos del inmueble y obtener la predicción del modelo entrenado.")

        # MAPEO: Relación entre ciudades (L3) y sus barrios (L4)
        # Aquí debes completar con tus datos reales
        mapeo_l3_l4 = {
            'Medellin': ['Alfonso López', 'Altavista', 'Aranjuez', 'Bellavista', 'Belén', 
                            'Buenos Aires', 'Calasanz', 'Calatrava', 'Campo Amor', 'Candelaria', 
                            'Castilla', 'Cristo Rey', 'Doce de Octubre', 'El Poblado', 'El Salado', 
                            'Estadio', 'Guayabal', 'La América', 'La Candelaria', 'Laureles', 
                            'Manrique', 'Robledo'],
            'Bello': ['Cabañas', 'Fontidueño', 'Niquía', 'La Frontera'],
            'Envigado': ['El Dorado', 'El Pedrero', 'La Magnolia', 'Zuniga'],
            'Itagui': ['Ditaires', 'Fátima', 'Loma de Los Bernal', 'Los Colores'],
            'Sabaneta': ['Holanda', 'Las Palmas', 'María Auxiliadora'],
            'La Estrella': ['La Tablaza', 'El Pedrero'],
            'Rionegro': ['El Porvenir', 'La Doctora', 'La Estación'],
            'Copacabana': ['La Pilarica', 'La Misericordia','Jardines'],
            'Caldas': ['La Floresta', 'Jardines'],
            'Retiro': ['Los Alpes', 'Los Lagos'],
            'Barbosa': ['Castropol', 'Kennedy'],
            'Girardota': ['Manila', 'Machado'],
            'La Ceja': ['Mayorca'],
            # Añade el resto de ciudades con sus barrios correspondientes
            # Si una ciudad no tiene barrios específicos en tu dataset, deja lista vacía
        }

        # Lista completa de todas las opciones L3 (ciudades) en orden alfabético
        opciones_l3 = [
            'Abejorral', 'Alejandría', 'Amalfi', 'Andes', 'Apartadó', 'Barbosa',
            'Bello', 'Betania', 'Caldas', 'Carepa', 'Caucasia', 'Chigorodó',
            'Ciudad Bolívar', 'Cocorná', 'Concepción', 'Concordia', 'Copacabana',
            'Ebéjico', 'El Carmen de Viboral', 'Envigado', 'Fredonia', 'Giraldo',
            'Girardota', 'Guarne', 'Guatapé', 'Hispania', 'Itagui', 'Jardín',
            'Jericó', 'La Ceja', 'La Estrella', 'La Pintada', 'La Unión',
            'Marinilla', 'Medellín', 'Necoclí', 'Olaya', 'Peñol', 'Puerto Triunfo',
            'Remedios', 'Retiro', 'Rionegro', 'Sabaneta', 'San Francisco',
            'San Jerónimo', 'San Pedro de los Milagros', 'San Rafael', 'San Roque',
            'San Vicente', 'Santafé de Antioquia', 'Segovia', 'Sopetrán',
            'Titiribí', 'Turbo', 'Urrao', 'Venecia', 'Yarumal'
        ]

        # Lista completa de todas las opciones L3 (ciudades)
        opciones_l3 = list(mapeo_l3_l4.keys())

        # --- ELEMENTOS FUERA DEL FORMULARIO para permitir actualización dinámica ---
        st.markdown("### 📍 Ubicación del Inmueble")

        col_a, col_b = st.columns(2)

        with col_a:
            # Selectbox de ciudad (L3) - FUERA del formulario
            localidad_l3_ej = st.selectbox(
                "Ciudad / Municipio (L3)",
                options=opciones_l3,
                index=opciones_l3.index("Medellín") if "Medellín" in opciones_l3 else 0,
                help="Selecciona la ciudad",
                key="select_l3"
            )

        with col_b:
            # Filtrar barrios según la ciudad seleccionada
            barrios_disponibles = mapeo_l3_l4.get(localidad_l3_ej, [])

            # Si no hay barrios específicos, mostrar mensaje
            if len(barrios_disponibles) == 0:
                st.info(f"ℹ No hay barrios específicos registrados para {localidad_l3_ej}")
                localidad_l4_ej = "Sin especificar"
                st.text_input(
                    "Sub-Barrio / Zona (L4)",
                    value="Sin especificar",
                    disabled=True,
                    key="select_l4_disabled"
                )
            else:
                # Selectbox de barrio (L4) - FUERA del formulario
                localidad_l4_ej = st.selectbox(
                    "Barrio / Zona (L4)",
                    options=barrios_disponibles,
                    index=0,
                    help="Selecciona el barrio (actualizado según la ciudad)",
                    key="select_l4"
                )

        # --- FORMULARIO para el resto de datos y el botón de predicción ---
        st.markdown("### 🏠 Características del Inmueble")

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                superficie_ej = st.number_input(
                    "Superficie Total (m²)",
                    min_value=0.0, max_value=10000.0, value=192.0, step=0.1, format="%.2f"
                )

            with col2:
                dormitorios_ej = st.number_input(
                    "Dormitorios",
                    min_value=0, max_value=20, value=5, step=1
                )

            with col3:
                banos_ej = st.number_input(
                    "Baños",
                    min_value=0, max_value=10, value=2, step=1
                )

            submit_button = st.form_submit_button(label="🔮 Predecir Precio", use_container_width=True)

        # --- Si el usuario envía el formulario ---
        if submit_button:
            st.markdown("---")
            st.subheader("🔍 Datos Ingresados")
            
            datos_usuario = pd.DataFrame({
                "Superficie (m²)": [superficie_ej],
                "Dormitorios": [dormitorios_ej],
                "Baños": [banos_ej],
                "Ciudad (L3)": [localidad_l3_ej],
                "Barrio (L4)": [localidad_l4_ej],
            })
            st.dataframe(datos_usuario, use_container_width=True)

            try:
                with st.spinner("Calculando predicción..."):
                    # Usamos el modelo cargado en sesión
                    modelo_cargado = st.session_state.best_model

                    # Llamamos a la función de predicción
                    precio_predicho = predecir_precio_vivienda(
                        superficie_ej,
                        dormitorios_ej,
                        banos_ej,
                        localidad_l3_ej,
                        localidad_l4_ej,
                        modelo_cargado
                    )

                # --- Mostrar resultados ---
                st.subheader("💰 Resultado de la Predicción")
                st.success(f"*Precio estimado en el año 2021: ${precio_predicho:,.2f}*")

                st.info(f"""
                *Resumen del Inmueble:*
                - 📐 Superficie Total: {superficie_ej} m²
                - 🛏 Dormitorios: {dormitorios_ej}
                - 🚿 Baños: {banos_ej}
                - 🏙 Ciudad: {localidad_l3_ej}
                - 🏘 Barrio: {localidad_l4_ej}
                """)

            except Exception as e:
                st.error(f"⚠ Ocurrió un error durante la predicción: {e}")

    else:
        st.warning("Por favor, carga los datos en la Pestaña 5 (entrenamiento) primero.")
with tab8:
    st.header("Paso 7: Interpretación del Modelo")
    
    if st.button("Interpretar Modelo"):
        with st.spinner("Interpretando el modelo..."):
            best_model = st.session_state.best_model
            model_name = st.session_state.model_name
            
            # Mostrar métricas del modelo
            st.subheader("Evaluación Final del Modelo Optimizado:")
            st.markdown("El modelo optimizado que en nuestro caso fue: **RandomForestRegressor** se evaluó en el conjunto de Prueba, datos que nunca se utilizaron en el entrenamiento o la validación.")
            st.text("")
            metrics_data = {
                "Métrica": ["R2", "MAE", "RMSE"],
                "Valor": [
                    "0.8436063284766978", 
                    "${:,.2f}".format(42487513.483914204), 
                    "${:,.2f}".format(5253572023866604.0)
                ]
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df)

            st.subheader("Predicción del Mejor Modelo")
            st.markdown("Se implementó una función para usar el modelo optimizado **(best_model)** y predecir el precio de una vivienda, simulando la entrada de datos de un usuario.")
            # Datos para la tabla
            data = {
                "Característica": [
                    "Superficie Total", 
                    "Dormitorios", 
                    "Baños", 
                    "Municipio (L3)", 
                    "Barrio (L4)"
                ],
                "Valor": [
                    "260.36 m²", 
                    "5", 
                    "1", 
                    "Copacabana", 
                    "Jardines"
                ]
            }

            # Crear un DataFrame
            df = pd.DataFrame(data)

            # Mostrar la tabla en Streamlit
            st.table(df)
            st.markdown("Resultado de la Predicción:")
            st.markdown("El Precio Predicho de la Vivienda es: **$421,392,604.28**")

                            

            st.markdown("""
            # Resumen del Proyecto
            
            ### **Valor de la Preparación de Datos**
            El proceso de limpieza, especialmente la **Extracción por Text Mining** para recuperar datos de área y ubicación, fue crucial. Esto demuestra que el **80% del éxito en Ciencia de Datos** radica en tener información completa y de alta calidad.

            ### **Impacto de la Ubicación**
            El uso de las variables categóricas **l3_final** (municipio) y **l4_final** (barrio) dentro del modelo, mediante **One-Hot Encoding**, permitió al algoritmo capturar el valor marginal de la localización, el cual es un factor determinante en el precio inmobiliario.

            ### **Modelo Robusto**
            La **Validación Cruzada** confirmó la solidez del modelo Random Forest frente a otros, asegurando que el rendimiento reportado no es un golpe de suerte, sino una métrica estable y confiable.
                        
            ### **Alto Poder Predictivo**
            El proyecto logró desarrollar un modelo (**Random Forest**) con un $R^2$ de **0.8436**, lo que demuestra una alta capacidad para predecir los precios de las propiedades en Antioquia basándose en las características físicas y de ubicación.

            """)



