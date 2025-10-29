#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo práctico de aprendizaje supervisado: Predicción de supervivencia en el Titanic
Este script implementa un flujo de trabajo completo de machine learning para predecir
qué pasajeros sobrevivieron al naufragio del Titanic.

Autor: Tania Rodriguez - Eder Lara
Fecha: 6 de junio de 2025
"""

# Importar bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
Bibliotecas Fundamentales
Estas son las librerías básicas para casi cualquier proyecto de ciencia de datos en Python.

import numpy as np

¿Qué hace? Es la librería fundamental para la computación numérica. Su principal objeto es el array multidimensional (ndarray), que es mucho más rápido y eficiente que las listas de Python para operaciones matemáticas.
¿Para qué se usa? Para todo tipo de cálculos matemáticos, álgebra lineal, transformaciones y manipulación de números a gran escala. Es la base sobre la que se construye Pandas.

import pandas as pd

¿Qué hace? Proporciona estructuras de datos de alto rendimiento y fáciles de usar, principalmente el DataFrame, que es como una tabla de Excel o una tabla de SQL dentro de Python.
¿Para qué se usa? Para leer, escribir, limpiar, filtrar, transformar, agrupar y analizar datos estructurados. Es la herramienta principal para la manipulación de datos. 🐼

import matplotlib.pyplot as plt

¿Qué hace? Es la librería de visualización más veterana y fundamental de Python. Te da un control total para crear una amplia variedad de gráficos estáticos, animados e interactivos.
¿Para qué se usa? Para crear gráficos básicos y personalizados como líneas, barras, histogramas y diagramas de dispersión.

import seaborn as sns

¿Qué hace? Es una librería de visualización basada en Matplotlib. Ofrece una interfaz de más alto nivel para crear gráficos estadísticos más atractivos y complejos con menos código.
¿Para qué se usa? Para crear visualizaciones estadísticas avanzadas como mapas de calor, diagramas de violín o gráficos de pares, que ayudan a explorar y entender las relaciones en los datos. 🎨
'''

# Para preprocesamiento
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

'''
Para Preprocesamiento
Estos módulos de scikit-learn se usan para preparar tus datos antes de entrenar un modelo.

from sklearn.model_selection import ...

    train_test_split: Divide tu conjunto de datos en dos partes: una para entrenar el modelo y otra para probar qué tan bien funciona con datos que nunca ha visto. Es un paso esencial para evitar el sobreajuste (overfitting).
    cross_val_score: Evalúa el modelo de forma más robusta mediante la validación cruzada. Divide los datos en múltiples "pliegues" (folds) y entrena/prueba el modelo varias veces, dándote un promedio de su rendimiento.
    GridSearchCV: Ayuda a encontrar los mejores hiperparámetros para un modelo. Prueba sistemáticamente una "rejilla" (grid) de combinaciones de parámetros y te dice cuál funcionó mejor.

from sklearn.preprocessing import ...

    StandardScaler: Estandariza las características numéricas para que tengan una media de 0 y una desviación estándar de 1. Es crucial para algoritmos sensibles a la escala de los datos, como las Máquinas de Soporte Vectorial (SVM).
    OneHotEncoder: Convierte variables categóricas (ej: "Rojo", "Verde", "Azul") en un formato numérico que el modelo pueda entender, creando nuevas columnas binarias (0s y 1s) para cada categoría.

from sklearn.impute import SimpleImputer

    ¿Qué hace? Maneja los valores faltantes (nulos o NaN) en tu dataset.
    ¿Para qué se usa? Para rellenar los datos faltantes con un valor específico, como la media, la mediana o la moda (el valor más frecuente) de la columna.

from sklearn.compose import ColumnTransformer

    ¿Qué hace? Permite aplicar diferentes transformaciones a diferentes columnas de tu dataset.
    ¿Para qué se usa? Es muy útil para, por ejemplo, aplicar StandardScaler a las columnas numéricas y OneHotEncoder a las columnas categóricas, todo en un solo paso.

from sklearn.pipeline import Pipeline

    ¿Qué hace? Encadena múltiples pasos de preprocesamiento y un modelo final en un solo objeto.
    ¿Para qué se usa? Para organizar el flujo de trabajo, evitar la fuga de datos (data leakage) y facilitar la aplicación de las mismas transformaciones a los datos de entrenamiento y prueba. Es como crear una línea de ensamblaje para tu modelo.
'''

# Para modelado
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

'''
Para Modelado
Estas son las clases que representan los algoritmos de Machine Learning que vas a entrenar.

from sklearn.linear_model import LogisticRegression

    Regresión Logística: A pesar de su nombre, es un modelo de clasificación. Es un algoritmo lineal simple pero potente, ideal como punto de partida para problemas de clasificación binaria (Sí/No).

from sklearn.tree import DecisionTreeClassifier

    Árbol de Decisión: Un modelo que aprende una serie de "preguntas" (reglas de decisión) para clasificar los datos. Es muy fácil de interpretar y visualizar.

from sklearn.ensemble import RandomForestClassifier

    Random Forest (Bosque Aleatorio): Un modelo de ensamble que construye muchos árboles de decisión y combina sus predicciones. Generalmente, es mucho más preciso y robusto que un solo árbol de decisión. 🌳

from sklearn.svm import SVC

    Support Vector Classifier (Máquina de Soporte Vectorial): Un modelo de clasificación muy potente que funciona encontrando el "hiperplano" que mejor separa las clases en los datos.
'''

# Para evaluación
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

'''
Para Evaluación
Estas funciones te ayudan a medir el rendimiento de tu modelo y a entender qué tan buenas son sus predicciones.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy_score (Exactitud): El porcentaje de predicciones correctas. (Correctas / Total).
    precision_score (Precisión): De todas las veces que el modelo predijo "Positivo", ¿cuántas acertó? Es importante cuando los falsos positivos son costosos.
    recall_score (Sensibilidad): De todos los casos que eran realmente "Positivos", ¿cuántos logró identificar el modelo? Es clave cuando los falsos negativos son peligrosos (ej: diagnóstico médico).
    f1_score: La media armónica de precisión y sensibilidad. Ofrece un buen balance entre ambas, especialmente útil cuando las clases están desbalanceadas.

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

    confusion_matrix (Matriz de Confusión): Una tabla que desglosa las predicciones en Verdaderos Positivos, Falsos Positivos, Verdaderos Negativos y Falsos Negativos. Es la base para calcular las demás métricas.
    classification_report (Reporte de Clasificación): Un resumen en texto que muestra la precisión, sensibilidad y F1-score para cada clase.
    roc_curve y roc_auc_score (Curva ROC y AUC): Herramientas para evaluar el rendimiento de un clasificador binario. La curva ROC visualiza el equilibrio entre la tasa de verdaderos positivos y falsos positivos, y el AUC (Área Bajo la Curva) resume este rendimiento en un solo número (1.0 es perfecto, 0.5 es aleatorio).

'''

# Configuración para visualizaciones
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Ignorar advertencias
import warnings
warnings.filterwarnings('ignore')

print("Entorno configurado correctamente.")

# Función para descargar los datos del Titanic
def cargar_datos():
    """
    Carga los datos del Titanic desde GitHub si no están disponibles localmente.
    
    Returns:
        pandas.DataFrame: Datos del Titanic
    """
    try:
        data = pd.read_csv('titanic.csv')
        print("Datos cargados correctamente desde archivo local.")
    except FileNotFoundError:
        print("Archivo no encontrado. Descargando datos...")
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        data = pd.read_csv(url)
        # Guardar localmente para uso futuro
        data.to_csv('titanic.csv', index=False)
        print("Datos descargados correctamente y guardados como 'titanic.csv'.")
    
    return data

# Función para explorar los datos
def explorar_datos(data):
    """
    Realiza un análisis exploratorio básico de los datos.
    
    Args:
        data (pandas.DataFrame): Datos del Titanic
    """
    print("\nPrimeras 5 filas del conjunto de datos:")
    print(data.head())
    
    print("\nInformación del conjunto de datos:")
    print(data.info())
    
    print("\nEstadísticas descriptivas:")
    print(data.describe())
    
    print("\nValores faltantes por columna:")
    print(data.isnull().sum())
    
    # Guardar visualizaciones en archivos
    # Distribución de la variable objetivo
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Survived', data=data)
    plt.title('Distribución de Supervivencia')
    plt.xlabel('Sobrevivió (1) / No Sobrevivió (0)')
    plt.ylabel('Cantidad de Pasajeros')
    plt.savefig('titanic_supervivencia.png', dpi=300, bbox_inches='tight')
    
    # Tasa de supervivencia por sexo
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Sex', y='Survived', data=data, ci=None)
    plt.title('Tasa de Supervivencia por Sexo')
    plt.xlabel('Sexo')
    plt.ylabel('Tasa de Supervivencia')
    plt.savefig('titanic_supervivencia_sexo.png', dpi=300, bbox_inches='tight')
    
    # Tasa de supervivencia por clase
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Pclass', y='Survived', data=data, ci=None)
    plt.title('Tasa de Supervivencia por Clase')
    plt.xlabel('Clase')
    plt.ylabel('Tasa de Supervivencia')
    plt.savefig('titanic_supervivencia_clase.png', dpi=300, bbox_inches='tight')
    
    # Distribución de edades
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='Age', hue='Survived', multiple='stack', bins=30)
    plt.title('Distribución de Edades por Supervivencia')
    plt.xlabel('Edad')
    plt.ylabel('Cantidad de Pasajeros')
    plt.legend(title='Sobrevivió', labels=['No', 'Sí'])
    plt.savefig('titanic_edad_supervivencia.png', dpi=300, bbox_inches='tight')
    
    print("\nVisualizaciones guardadas como archivos PNG.")
    
    # Crear una característica de tamaño de familia
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1  # +1 para incluir al pasajero
    
    # Tasa de supervivencia por tamaño de familia
    plt.figure(figsize=(10, 6))
    sns.barplot(x='FamilySize', y='Survived', data=data, ci=None)
    plt.title('Tasa de Supervivencia por Tamaño de Familia')
    plt.xlabel('Tamaño de Familia')
    plt.ylabel('Tasa de Supervivencia')
    plt.savefig('titanic_familia_supervivencia.png', dpi=300, bbox_inches='tight')
    
    return data

# Función para preparar los datos
def preparar_datos(data):
    """
    Prepara los datos para el modelado, incluyendo selección de características,
    división en conjuntos de entrenamiento y prueba, y creación de pipelines de preprocesamiento.
    
    Args:
        data (pandas.DataFrame): Datos del Titanic
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    # Seleccionar características relevantes
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = data[features]
    y = data['Survived']
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")
    
    # Identificar tipos de columnas
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    
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
    
    # Verificar la forma de los datos después del preprocesamiento
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    print(f"Forma de X_train después del preprocesamiento: {X_train_preprocessed.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor

# Función para entrenar y evaluar modelos
def entrenar_evaluar_modelos(X_train, X_test, y_train, y_test, preprocessor):
    """
    Entrena varios modelos y evalúa su rendimiento.
    
    Args:
        X_train, X_test, y_train, y_test: Conjuntos de datos de entrenamiento y prueba
        preprocessor: Transformador de columnas para preprocesamiento
        
    Returns:
        tuple: Mejor modelo, nombre del mejor modelo, resultados de todos los modelos
    """
    # Definir modelos a evaluar
    models = {
        'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
        'Árbol de Decisión': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Crear pipelines para cada modelo
    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    
    # Evaluar modelos con validación cruzada
    results = {}
    for name, pipeline in pipelines.items():
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        print(f"{name}: Exactitud CV = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Visualizar resultados de validación cruzada
    cv_means = [results[name]['cv_mean'] for name in models.keys()]
    cv_stds = [results[name]['cv_std'] for name in models.keys()]
    
    plt.figure(figsize=(12, 6))
    plt.bar(models.keys(), cv_means, yerr=cv_stds, capsize=10)
    plt.title('Comparación de Modelos (Validación Cruzada)')
    plt.xlabel('Modelo')
    plt.ylabel('Exactitud Media')
    plt.ylim([0.7, 0.9])  # Ajustar según los resultados
    plt.grid(axis='y')
    plt.savefig('titanic_comparacion_modelos.png', dpi=300, bbox_inches='tight')
    
    # Seleccionar el mejor modelo basado en validación cruzada
    best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
    best_pipeline = pipelines[best_model_name]
    print(f"\nMejor modelo: {best_model_name} con exactitud CV de {results[best_model_name]['cv_mean']:.4f}")
    
    # Entrenar el mejor modelo en todo el conjunto de entrenamiento
    best_pipeline.fit(X_train, y_train)
    
    # Evaluar en el conjunto de prueba
    y_pred = best_pipeline.predict(X_test)
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    # Métricas de rendimiento
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nRendimiento en el conjunto de prueba:")
    print(f"Exactitud: {accuracy:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Exhaustividad: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.savefig('titanic_matriz_confusion.png', dpi=300, bbox_inches='tight')
    
    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('titanic_curva_roc.png', dpi=300, bbox_inches='tight')
    
    # Informe de clasificación detallado
    print("\nInforme de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    return best_pipeline, best_model_name, results

# Función para optimizar hiperparámetros
def optimizar_hiperparametros(best_pipeline, best_model_name, X_train, y_train, X_test, y_test):
    """
    Optimiza los hiperparámetros del mejor modelo.
    
    Args:
        best_pipeline: Pipeline del mejor modelo
        best_model_name: Nombre del mejor modelo
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
        
    Returns:
        object: Modelo optimizado
    """
    # Definir espacio de búsqueda de hiperparámetros según el mejor modelo
    if best_model_name == 'Regresión Logística':
        param_grid = {
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__solver': ['liblinear', 'lbfgs'],
            'model__penalty': ['l1', 'l2']
        }
    elif best_model_name == 'Árbol de Decisión':
        param_grid = {
            'model__max_depth': [None, 5, 10, 15, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'Random Forest':
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'SVM':
        param_grid = {
            'model__C': [0.1, 1, 10, 100],
            'model__gamma': ['scale', 'auto', 0.1, 0.01],
            'model__kernel': ['rbf', 'linear']
        }
    
    # Realizar búsqueda en cuadrícula
    print("\nIniciando optimización de hiperparámetros. Esto puede tomar un tiempo...")
    grid_search = GridSearchCV(
        best_pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Mejores hiperparámetros
    print("\nMejores hiperparámetros:")
    print(grid_search.best_params_)
    print(f"Mejor puntuación de validación cruzada: {grid_search.best_score_:.4f}")
    
    # Evaluar modelo optimizado en el conjunto de prueba
    best_model = grid_search.best_estimator_
    y_pred_optimized = best_model.predict(X_test)
    y_pred_proba_optimized = best_model.predict_proba(X_test)[:, 1]
    
    # Métricas de rendimiento del modelo optimizado
    accuracy_opt = accuracy_score(y_test, y_pred_optimized)
    precision_opt = precision_score(y_test, y_pred_optimized)
    recall_opt = recall_score(y_test, y_pred_optimized)
    f1_opt = f1_score(y_test, y_pred_optimized)
    roc_auc_opt = roc_auc_score(y_test, y_pred_proba_optimized)
    
    print("\nRendimiento del modelo optimizado en el conjunto de prueba:")
    print(f"Exactitud: {accuracy_opt:.4f}")
    print(f"Precisión: {precision_opt:.4f}")
    print(f"Exhaustividad: {recall_opt:.4f}")
    print(f"F1-Score: {f1_opt:.4f}")
    print(f"AUC-ROC: {roc_auc_opt:.4f}")
    
    return best_model

# Función para guardar el modelo
def guardar_modelo(model, filename):
    """
    Guarda un modelo entrenado en un archivo usando joblib.
    
    Args:
        model (object): El modelo entrenado que se va a guardar.
        filename (str): El nombre del archivo (ej. 'modelo.joblib').
    """
    try:
        joblib.dump(model, filename)
        print(f"\nModelo guardado exitosamente en el archivo: '{filename}'")
    except Exception as e:
        print(f"\nError al guardar el modelo: {e}")

# Función para interpretar el modelo
def interpretar_modelo(model, best_model_name, X_test, y_test, preprocessor):
    """
    Interpreta el modelo final para entender qué características son más importantes.
    
    Args:
        model: Modelo optimizado
        best_model_name: Nombre del mejor modelo
        X_test, y_test: Datos de prueba
        preprocessor: Transformador de columnas para preprocesamiento
    """
    # Extraer importancia de características (si el modelo lo permite)
    if best_model_name in ['Árbol de Decisión', 'Random Forest']:
        # Para árboles, podemos obtener la importancia directamente
        numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
        categorical_features = ['Pclass', 'Sex', 'Embarked']
        
        # Obtener nombres de características después de one-hot encoding
        ohe = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(categorical_features)
        
        # Combinar nombres de características
        feature_names = np.array(numeric_features + list(cat_feature_names))
        
        # Obtener importancias
        importances = model.named_steps['model'].feature_importances_
        
        # Crear DataFrame para visualización
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Ordenar por importancia
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Visualizar
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Importancia de Características')
        plt.xlabel('Importancia')
        plt.ylabel('Característica')
        plt.grid(axis='x')
        plt.savefig('titanic_importancia_caracteristicas.png', dpi=300, bbox_inches='tight')
        
    elif best_model_name == 'Regresión Logística':
        # Para regresión logística, podemos obtener los coeficientes
        numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
        categorical_features = ['Pclass', 'Sex', 'Embarked']
        
        # Obtener nombres de características después de one-hot encoding
        ohe = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(categorical_features)
        
        # Combinar nombres de características
        feature_names = np.array(numeric_features + list(cat_feature_names))
        
        # Obtener coeficientes
        coefficients = model.named_steps['model'].coef_[0]
        
        # Crear DataFrame para visualización
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        })
        
        # Ordenar por valor absoluto de coeficientes
        feature_importance['AbsCoefficient'] = np.abs(feature_importance['Coefficient'])
        feature_importance = feature_importance.sort_values('AbsCoefficient', ascending=False)
        
        # Visualizar
        plt.figure(figsize=(12, 8))
        colors = ['red' if c < 0 else 'green' for c in feature_importance['Coefficient']]
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance, palette=colors)
        plt.title('Coeficientes de Regresión Logística')
        plt.xlabel('Coeficiente')
        plt.ylabel('Característica')
        plt.grid(axis='x')
        plt.savefig('titanic_coeficientes.png', dpi=300, bbox_inches='tight')
    
    # Análisis de errores
    y_pred_final = model.predict(X_test)
    errors = y_test != y_pred_final
    
    # Crear DataFrame con los datos de prueba y resultados
    X_test_reset = X_test.reset_index(drop=True)
    error_analysis = pd.DataFrame({
        'Real': y_test.reset_index(drop=True),
        'Predicción': y_pred_final,
        'Error': errors.reset_index(drop=True),
        'Probabilidad': model.predict_proba(X_test)[:, 1]
    })
    
    # Combinar con características originales
    for col in X_test.columns:
        error_analysis[col] = X_test_reset[col]
    
    # Mostrar ejemplos de errores
    print("\nEjemplos de predicciones incorrectas:")
    print(error_analysis[error_analysis['Error']].head())
    
    # Analizar errores por características
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Sex', hue='Error', data=error_analysis)
    plt.title('Distribución de Errores por Sexo')
    plt.xlabel('Sexo')
    plt.ylabel('Cantidad')
    plt.legend(title='Error', labels=['Correcto', 'Incorrecto'])
    plt.savefig('titanic_errores_sexo.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Pclass', hue='Error', data=error_analysis)
    plt.title('Distribución de Errores por Clase')
    plt.xlabel('Clase')
    plt.ylabel('Cantidad')
    plt.legend(title='Error', labels=['Correcto', 'Incorrecto'])
    plt.savefig('titanic_errores_clase.png', dpi=300, bbox_inches='tight')
    
    # Análisis de errores por edad
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Error', y='Age', data=error_analysis)
    plt.title('Distribución de Edades por Error')
    plt.xlabel('Error')
    plt.ylabel('Edad')
    plt.xticks([0, 1], ['Correcto', 'Incorrecto'])
    plt.savefig('titanic_errores_edad.png', dpi=300, bbox_inches='tight')

# Función para cargar el modelo
def cargar_modelo(filename):
    """
    Carga un modelo guardado desde un archivo joblib.
    
    Args:
        filename (str): La ruta al archivo .joblib del modelo.
        
    Returns:
        object: El modelo cargado, o None si ocurre un error.
    """
    try:
        model = joblib.load(filename)
        print(f"\nModelo '{filename}' cargado exitosamente.")
        return model
    except FileNotFoundError:
        print(f"\nError: No se encontró el archivo del modelo en '{filename}'")
        return None
    except Exception as e:
        print(f"\nOcurrió un error al cargar el modelo: {e}")
        return None

# Función para hacer predicciones con nuevos datos
def hacer_prediccion(model, new_data):
    """
    Hace predicciones con nuevos datos.
    
    Args:
        model: Modelo entrenado
        new_data: Nuevos datos para predecir
        
    Returns:
        array: Predicciones
    """
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)[:, 1]
    
    results = pd.DataFrame({
        'Predicción': predictions,
        'Probabilidad de Supervivencia': probabilities
    })
    
    return results

# Función principal
def main():
    """
    Función principal que ejecuta todo el flujo de trabajo.
    """
    print("Iniciando análisis de supervivencia en el Titanic...")
    
    # Cargar datos
    data = cargar_datos()
    
    # Explorar datos
    data = explorar_datos(data)
    
    # Preparar datos
    X_train, X_test, y_train, y_test, preprocessor = preparar_datos(data)
    
    # Entrenar y evaluar modelos
    best_pipeline, best_model_name, results = entrenar_evaluar_modelos(X_train, X_test, y_train, y_test, preprocessor)
    
    # Optimizar hiperparámetros
    best_model = optimizar_hiperparametros(best_pipeline, best_model_name, X_train, y_train, X_test, y_test)

    # Guardar el modelo optimizado
    if best_model:
        guardar_modelo(best_model, 'titanic_survival_model.joblib')

    # Interpretar modelo
    interpretar_modelo(best_model, best_model_name, X_test, y_test, preprocessor)
    
    # Ejemplo de predicción con nuevos datos
    print("\nEjemplo de predicción con nuevos pasajeros:")
    
    # Cargamos el modelo desde el disco
    modelo_produccion = cargar_modelo('titanic_survival_model.joblib')
    
    if modelo_produccion:
        # Crear algunos pasajeros de ejemplo
        new_passengers = pd.DataFrame({
            'Pclass': [1, 3, 2],
            'Sex': ['female', 'male', 'female'],
            'Age': [29, 35, 10],
            'SibSp': [0, 1, 1],
            'Parch': [0, 0, 1],
            'Fare': [100, 15, 30],
            'Embarked': ['S', 'S', 'C']
        })
        
        print("\nNuevos pasajeros:")
        print(new_passengers)
        
        # Hacer predicciones con el modelo cargado
        predictions = hacer_prediccion(modelo_produccion, new_passengers)
        
        # Mostrar resultados
        result_df = pd.concat([new_passengers, predictions], axis=1)
        print("\nResultados de predicción:")
        print(result_df)

    print("\nAnálisis completado.")
    
    # Mostrar resultados
    result_df = pd.concat([new_passengers, predictions], axis=1)
    print("\nResultados de predicción:")
    print(result_df)
    
    print("\nAnálisis completado. Todas las visualizaciones han sido guardadas como archivos PNG.")

# Ejecutar el programa si se llama directamente
if __name__ == "__main__":
    main()

