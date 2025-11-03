import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
import math




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
    # Guardar la matriz de correlación como una imagen
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlaciones, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.savefig('matriz_correlacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    return correlaciones

def graficar_distribucion_numericas(data, columnas_numericas):
    img_path = "boxplots_numericas.png"
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

    return results

def preparar_datos():
    # Construye la ruta absoluta al archivo CSV
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, '..', 'data', 'properties_gold.csv')
    try:
        data = pd.read_csv(file_path)
        # Filtrar filas con valores nulos en columnas críticas
        data= data.dropna(subset=['created_on', 'price', 'surface_total_final', 'bedrooms_final', 'bathrooms_final', 'l3_final', 'l4_final']).copy()        
        data.to_csv('../data/properties_final.csv', index=False)
        print("Datos cargados correctamente desde archivo local.")
    except FileNotFoundError:
        print("Archivo no encontrado.")
        # Aquí podrías descargar o crear un DataFrame vacío
        data = pd.DataFrame()
    return data




