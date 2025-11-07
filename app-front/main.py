import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
import math
import matplotlib.pyplot as plt




def cargar_datos():
    # Construye la ruta absoluta al archivo CSV
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, '..', 'data', 'properties_antioquia.csv')
    try:
        data = pd.read_csv(file_path)
        print("Datos cargados correctamente desde archivo local.")
    except FileNotFoundError:
        print("Archivo no encontrado.")
        # Aquí podrías descargar o crear un DataFrame vacío
        data = pd.DataFrame()
    return data



def identificar_tipos_columnas(dataframe):
    numericas = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categoricas = dataframe.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numericas, categoricas

def correlacion_variables(data, columnas_numericas):
    """
    Calcula y visualiza la matriz de correlación para las variables numéricas especificadas.
    """
    correlaciones = data[columnas_numericas].corr()

    # Definir la nueva ruta para guardar las imágenes
    base_path = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_path, '..', 'plots')

    # Crear el directorio si no existe
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    img_path = os.path.join(plots_dir, "matriz_correlacion.png")

    # Guardar la matriz de correlación como una imagen
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlaciones, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Matriz de correlación guardada como {img_path}")
    return correlaciones

def graficar_distribucion_numericas(data, columnas_numericas):
    # Definir la nueva ruta para guardar las imágenes
    base_path = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_path, '..', 'plots')

    # Crear el directorio si no existe
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    img_path = os.path.join(plots_dir, "boxplots_numericas.png")

    if not os.path.exists(img_path):
        n = len(columnas_numericas)
        ncols = 2 if n <= 4 else 3
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten() if n > 1 else [axes]
        for i, column in enumerate(columnas_numericas):
            if column in data.columns:
                sns.boxplot(x=data[column].dropna(), ax=axes[i])
                axes[i].set_title(f'Boxplot de {column}')
            else:
                axes[i].set_visible(False)
        # Oculta los ejes no usados
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        fig.savefig(img_path)
        plt.close(fig)



def graficar_barrios(data):
    columna = 'l4'  # Columna fija para graficar los barrios

    if columna not in data.columns:
        print(f"Advertencia: La columna '{columna}' no existe en el DataFrame.")
        return

    # Definir la nueva ruta para guardar las imágenes
    base_path = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_path, '..', 'plots')

    # Crear el directorio si no existe
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    img_path = os.path.join(plots_dir, "grafico_barrios_l4.png")

    # Verificar si la imagen ya existe
    if os.path.exists(img_path):
        print(f"El gráfico ya existe: {img_path}")
        return

    # Contar la cantidad de datos por barrio
    conteo_barrios = data[columna].value_counts()

    # Crear el gráfico de barras
    plt.figure(figsize=(14, 10))  # Aumentar el tamaño del gráfico para más espacio
    ax = conteo_barrios.plot(kind='barh', color='skyblue')  # Cambiar a gráfico de barras horizontales
    plt.title('Cantidad de datos por barrio', fontsize=16)
    plt.xlabel('Cantidad de datos', fontsize=14)
    plt.ylabel('Barrios', fontsize=14, labelpad=20, rotation=0)  # Mantener el ylabel horizontal
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Agregar el conteo al final de cada barra
    for i, v in enumerate(conteo_barrios):
        ax.text(v + 1, i, str(v), color='black', va='center', fontsize=12)

    # Guardar el gráfico como una imagen
    plt.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Gráfico guardado como {img_path}")

def explorar_datos(data):
    """
    Realiza un análisis exploratorio básico de los datos y retorna resultados como tablas.
    """
    results = {}
    #results["Primeras 5 filas"] = data.head()
    #results["Info"] = data.info()
    results["Cantidad de registros"] = data.shape
    results["Estadísticas descriptivas"] = data.describe()
    results["Valores faltantes por columna"] = data.isnull().sum()

    # Función para determinar columnas numéricas y categóricas

    results["Columnas numéricas"], results["Columnas categóricas"] = identificar_tipos_columnas(data)

    # Graficar distribución de variables numéricas
    columnas_a_graficar = ['bedrooms', 'bathrooms', 'price', 'surface_total']
    graficar_distribucion_numericas(data, columnas_a_graficar)

    # Usar solo columnas numéricas para la correlación
    columnas = ['bedrooms', 'bathrooms', 'price', 'surface_total', 'surface_covered','surface_total','rooms','lat','lon']
    correlacion_variables(data, columnas)

    # histograma de la variable objetivo 'l4_final'
    graficar_barrios(data)

    return results

def preparar_datos():
    # Construye la ruta absoluta al archivo CSV
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, '..', 'data', 'properties_final.csv')
    try:
        data = pd.read_csv(file_path)       
        print("Datos cargados correctamente desde archivo local.")
    except FileNotFoundError:
        print("Archivo no encontrado.")
        # Aquí podrías descargar o crear un DataFrame vacío
        data = pd.DataFrame()
    """#Eliminación de outliers de precio"""
    Q1 = data['price'].quantile(0.25)
    Q3 = data['price'].quantile(0.75)
    IQR = Q3 - Q1

    # Filtrar datos dentro de 1.5*IQR del rango intercuartílico
    data = data[(data['price'] >= Q1 - 1.5 * IQR) & (data['price'] <= Q3 + 1.5 * IQR)]

    # Gráfico de dispersión de precios",
    plots_dir = os.path.join(base_path, '..', 'plots')

    # Crear el directorio si no existe
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    img_path = os.path.join(plots_dir, "boxplot_precios_sin_outliers.png")

    # Verificar si la imagen ya existe
    if not os.path.exists(img_path):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data['price'].dropna())
        plt.title('Boxplot de precios de propiedades')
        plt.xlabel('Precio')
        plt.grid(True)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Boxplot guardado como {img_path}")
    else:
        print(f"El boxplot ya existe: {img_path}")

    # Definición del límite
    LIMITE_SUPERIOR = 2000


    # 1. Aplicar la eliminación de valores superiores
    # La condición mantiene los valores <= 1000
    data = data[data['surface_total_final'] <= LIMITE_SUPERIOR]

    # 2. Resumen de la eliminación
    registros_eliminados = data.shape[0] - data.shape[0]

    print(f"--- Eliminación de {'surface_total_final'} ---")
    print(f"Límite aplicado: {'surface_total_final'} > {LIMITE_SUPERIOR}")
    print(f"Tamaño original del DataFrame: {data.shape[0]} filas")
    print(f"Tamaño del DataFrame después de filtrar: {data.shape[0]} filas")
    print(f"Registros eliminados: {registros_eliminados} filas")

    # 3. Gráfico de dispersión para verificar el resultado
    plots_dir = os.path.join(base_path, '..', 'plots')

    # Crear el directorio si no existe
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    img_path = os.path.join(plots_dir, "dispersión_surface_total_final.png")

    # Verificar si la imagen ya existe
    if not os.path.exists(img_path):
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(data)), data['surface_total_final'], alpha=0.6)
        plt.title(f'Dispersión de surface_total_final (Valores > {LIMITE_SUPERIOR} Eliminados)')
        plt.xlabel('Índice de muestra')
        plt.ylabel('surface_total_final')
        plt.grid(True)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Gráfico de dispersión guardado como {img_path}")
    else:
        print(f"El gráfico de dispersión ya existe: {img_path}")
           


    return data

def entrenar_modelo():








