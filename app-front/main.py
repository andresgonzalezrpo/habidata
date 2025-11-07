import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Para preprocesamiento
from sklearn.model_selection import train_test_split, GridSearchCV,cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Para modelado
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR





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

def preparar_modelo(data):
    features = ['created_on','surface_total_final', 'bedrooms_final', 'bathrooms_final', 'l3_final', 'l4_final']
    X = data[features]
    y = data['price']

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   
    # Identificar tipos de columnas
    numeric_features = ['surface_total_final', 'bedrooms_final', 'bathrooms_final']
    categorical_features = ['l3_final', 'l4_final']

    # Crear transformadores para diferentes tipos de columnas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combinar transformadores usando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

def entrenar_evaluar_modelos(X_train, X_test, y_train, y_test, preprocessor):
    """
    Entrena varios modelos y evalúa su rendimiento.
    
    Args:
        X_train, X_test, y_train, y_test: Conjuntos de datos de entrenamiento y prueba
        preprocessor: Transformador de columnas para preprocesamiento
        
    Returns:
        tuple: Mejor modelo, nombre del mejor modelo, resultados de todos los modelos
    """

    # Verificar la forma de los datos después del preprocesamiento
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    print(f"Forma de X_train después del preprocesamiento: {X_train_preprocessed.shape}")

    models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Support Vector Regressor": SVR()
    }

    """#Validación cruzada, comparación y resultados"""

    # results = {}
    # for name, model in models.items():
    #     pipeline = Pipeline(steps=[('preprocessor', preprocessor),
    #                              ('model', model)])
    #     scores = cross_validate(pipeline, X_train, y_train,
    #                             cv=5,
    #                             scoring=('r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'),
    #                             return_train_score=True)
    #     results[name] = {
    #         'R2_mean': np.mean(scores['test_r2']),
    #         'MAE_mean': -np.mean(scores['test_neg_mean_absolute_error']),
    #         'RMSE_mean': -np.mean(scores['test_neg_root_mean_squared_error'])
    #     }

    # cv_results = pd.DataFrame(results).T.sort_values(by='R2_mean', ascending=False)
    # print(cv_results)

    # print("\nGenerando gráficos de resultados de CV...")
    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    # plt.subplots_adjust(hspace=0.5) # Aumentar espacio entre subgráficos

    # # Gráfico 1: R2_mean (Mayor es mejor)
    # cv_results['R2_mean'].plot(kind='bar', ax=axes[0], color='skyblue')
    # axes[0].set_title('R2 Promedio de Validación Cruzada (Mayor es Mejor)', fontsize=14)
    # axes[0].set_ylabel('R2 Promedio')
    # axes[0].set_xlabel('Modelo')
    # axes[0].tick_params(axis='x', rotation=45)
    # axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # # Gráfico 2: MAE_mean y RMSE_mean (Menor es mejor)
    # cv_results[['MAE_mean', 'RMSE_mean']].plot(kind='bar', ax=axes[1])
    # axes[1].set_title('Error Promedio de Validación Cruzada (Menor es Mejor)', fontsize=14)
    # axes[1].set_ylabel('Valor de Error')
    # axes[1].set_xlabel('Modelo')

    # # Usamos formato simple para las etiquetas del eje Y para evitar notación científica grande
    # axes[1].ticklabel_format(style='plain', axis='y') 
    # axes[1].tick_params(axis='x', rotation=45)
    # axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # plt.tight_layout()
    # plt.show()


    # """#Entrenar modelo final y evaluar en test con el mejor modelo que fue RandomForestRegressor:"""

    best_model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', RandomForestRegressor(random_state=42))])

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # print("R2:", r2_score(y_test, y_pred))
    # print("MAE:", mean_absolute_error(y_test, y_pred))
    # print("RMSE:", mean_squared_error(y_test, y_pred))

    return best_model



    # # Definir modelos a evaluar
    # models = {
    #     'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
    #     'Árbol de Decisión': DecisionTreeClassifier(random_state=42),
    #     'Random Forest': RandomForestClassifier(random_state=42),
    #     'SVM': SVC(probability=True, random_state=42)
    # }
    
    # # Crear pipelines para cada modelo
    # pipelines = {}
    # for name, model in models.items():
    #     pipelines[name] = Pipeline(steps=[
    #         ('preprocessor', preprocessor),
    #         ('model', model)
    #     ])
    
    # # Evaluar modelos con validación cruzada
    # results = {}
    # for name, pipeline in pipelines.items():
    #     cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    #     results[name] = {
    #         'cv_mean': cv_scores.mean(),
    #         'cv_std': cv_scores.std()
    #     }
    #     print(f"{name}: Exactitud CV = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # # Visualizar resultados de validación cruzada
    # cv_means = [results[name]['cv_mean'] for name in models.keys()]
    # cv_stds = [results[name]['cv_std'] for name in models.keys()]
    
    # plt.figure(figsize=(12, 6))
    # plt.bar(models.keys(), cv_means, yerr=cv_stds, capsize=10)
    # plt.title('Comparación de Modelos (Validación Cruzada)')
    # plt.xlabel('Modelo')
    # plt.ylabel('Exactitud Media')
    # plt.ylim([0.7, 0.9])  # Ajustar según los resultados
    # plt.grid(axis='y')
    # plt.savefig('titanic_comparacion_modelos.png', dpi=300, bbox_inches='tight')
    
    # # Seleccionar el mejor modelo basado en validación cruzada
    # best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
    # best_pipeline = pipelines[best_model_name]
    # print(f"\nMejor modelo: {best_model_name} con exactitud CV de {results[best_model_name]['cv_mean']:.4f}")
    
    # # Entrenar el mejor modelo en todo el conjunto de entrenamiento
    # best_pipeline.fit(X_train, y_train)
    
    # # Evaluar en el conjunto de prueba
    # y_pred = best_pipeline.predict(X_test)
    # y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    # # Métricas de rendimiento
    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    
    # print("\nRendimiento en el conjunto de prueba:")
    # print(f"Exactitud: {accuracy:.4f}")
    # print(f"Precisión: {precision:.4f}")
    # print(f"Exhaustividad: {recall:.4f}")
    # print(f"F1-Score: {f1:.4f}")
    
    # # Matriz de confusión
    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.title('Matriz de Confusión')
    # plt.xlabel('Predicción')
    # plt.ylabel('Valor Real')
    # plt.savefig('titanic_matriz_confusion.png', dpi=300, bbox_inches='tight')
    
    # # Curva ROC
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    # roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # plt.figure(figsize=(10, 8))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Tasa de Falsos Positivos')
    # plt.ylabel('Tasa de Verdaderos Positivos')
    # plt.title('Curva ROC')
    # plt.legend(loc="lower right")
    # plt.grid(True)
    # plt.savefig('titanic_curva_roc.png', dpi=300, bbox_inches='tight')
    
    # # Informe de clasificación detallado
    # print("\nInforme de Clasificación:")
    # print(classification_report(y_test, y_pred))
    
    # return best_pipeline, best_model_name, results
    



    
