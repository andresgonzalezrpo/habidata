import pandas as pd
import os
import io




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

def explorar_datos(data):
    """
    Realiza un análisis exploratorio básico de los datos y retorna resultados como tablas.
    """
    results = {}
    results["Primeras 5 filas"] = data.head()
    # data.info() prints to stdout, so capture it as a string
    
    buffer = io.StringIO()
    data.info(buf=buffer)
    results["Info"] = buffer.getvalue()
    results["Estadísticas descriptivas"] = data.describe()
    results["Valores faltantes por columna"] = data.isnull().sum()
    return results    

    