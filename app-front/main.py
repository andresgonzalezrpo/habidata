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

# def explorar_datos(data):
#     """
#     Realiza un análisis exploratorio básico de los datos.
    
#     Args:
#         data (pandas.DataFrame): Datos del Titanic
#     """
#     print("\nPrimeras 5 filas del conjunto de datos:")
#     print(data.head())
    
#     print("\nInformación del conjunto de datos:")
#     print(data.info())
    
#     print("\nEstadísticas descriptivas:")
#     print(data.describe())
    
#     print("\nValores faltantes por columna:")
#     print(data.isnull().sum())
    
    # # Guardar visualizaciones en archivos
    # # Distribución de la variable objetivo
    # plt.figure(figsize=(8, 6))
    # sns.countplot(x='Survived', data=data)
    # plt.title('Distribución de Supervivencia')
    # plt.xlabel('Sobrevivió (1) / No Sobrevivió (0)')
    # plt.ylabel('Cantidad de Pasajeros')
    # plt.savefig('titanic_supervivencia.png', dpi=300, bbox_inches='tight')
    
    # # Tasa de supervivencia por sexo
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='Sex', y='Survived', data=data, ci=None)
    # plt.title('Tasa de Supervivencia por Sexo')
    # plt.xlabel('Sexo')
    # plt.ylabel('Tasa de Supervivencia')
    # plt.savefig('titanic_supervivencia_sexo.png', dpi=300, bbox_inches='tight')
    
    # # Tasa de supervivencia por clase
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='Pclass', y='Survived', data=data, ci=None)
    # plt.title('Tasa de Supervivencia por Clase')
    # plt.xlabel('Clase')
    # plt.ylabel('Tasa de Supervivencia')
    # plt.savefig('titanic_supervivencia_clase.png', dpi=300, bbox_inches='tight')
    
    # # Distribución de edades
    # plt.figure(figsize=(12, 6))
    # sns.histplot(data=data, x='Age', hue='Survived', multiple='stack', bins=30)
    # plt.title('Distribución de Edades por Supervivencia')
    # plt.xlabel('Edad')
    # plt.ylabel('Cantidad de Pasajeros')
    # plt.legend(title='Sobrevivió', labels=['No', 'Sí'])
    # plt.savefig('titanic_edad_supervivencia.png', dpi=300, bbox_inches='tight')
    
    # print("\nVisualizaciones guardadas como archivos PNG.")
    
    # # Crear una característica de tamaño de familia
    # data['FamilySize'] = data['SibSp'] + data['Parch'] + 1  # +1 para incluir al pasajero
    
    # # Tasa de supervivencia por tamaño de familia
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='FamilySize', y='Survived', data=data, ci=None)
    # plt.title('Tasa de Supervivencia por Tamaño de Familia')
    # plt.xlabel('Tamaño de Familia')
    # plt.ylabel('Tasa de Supervivencia')
    # plt.savefig('titanic_familia_supervivencia.png', dpi=300, bbox_inches='tight')

    # return data

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



