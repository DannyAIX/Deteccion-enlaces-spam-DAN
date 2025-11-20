import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PASO 1: CARGA DEL CONJUNTO DE DATOS
# ============================================
print("=" * 60)
print("PASO 1: Cargando el conjunto de datos...")
print("=" * 60)

url = "https://breathecode.herokuapp.com/asset/internal-link?id=932&path=url_spam.csv"
df = pd.read_csv(url)

print(f"\nDimensiones del dataset: {df.shape}")
print(f"\nPrimeras filas del dataset:")
print(df.head())
print(f"\nInformación del dataset:")
print(df.info())
print(f"\nDistribución de clases:")
print(df['is_spam'].value_counts())
print(f"\nProporción de spam: {df['is_spam'].mean():.2%}")

# ============================================
# PASO 2: PREPROCESAMIENTO DE LOS ENLACES
# ============================================
print("\n" + "=" * 60)
print("PASO 2: Preprocesando los enlaces...")
print("=" * 60)

def preprocess_url(url):
    """
    Preprocesa una URL para extraer características útiles
    """
    if pd.isna(url):
        return ""
    
    # Convertir a minúsculas
    url = url.lower()
    
    # Eliminar el protocolo (http://, https://, etc.)
    url = re.sub(r'https?://', '', url)
    url = re.sub(r'www\.', '', url)
    
    # Separar por caracteres especiales comunes en URLs
    # Esto ayuda a tokenizar la URL en partes significativas
    url = re.sub(r'[/\-._=?&]', ' ', url)
    
    # Eliminar números (opcional, pero puede ayudar)
    # url = re.sub(r'\d+', '', url)
    
    # Eliminar espacios extras
    url = ' '.join(url.split())
    
    return url

# Aplicar preprocesamiento
df['url_processed'] = df['url'].apply(preprocess_url)

print("\nEjemplos de URLs procesadas:")
for i in range(5):
    print(f"\nOriginal: {df['url'].iloc[i]}")
    print(f"Procesada: {df['url_processed'].iloc[i]}")
    print(f"Es spam: {df['is_spam'].iloc[i]}")

# ============================================
# División del conjunto de datos
# ============================================
print("\n" + "=" * 60)
print("Dividiendo el dataset en train y test...")
print("=" * 60)

X = df['url_processed']
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\nTamaño del conjunto de entrenamiento: {len(X_train)}")
print(f"Tamaño del conjunto de prueba: {len(X_test)}")
print(f"\nDistribución en train: {y_train.value_counts().to_dict()}")
print(f"Distribución en test: {y_test.value_counts().to_dict()}")

# ============================================
# PASO 3: CONSTRUIR UN SVM CON PARÁMETROS POR DEFECTO
# ============================================
print("\n" + "=" * 60)
print("PASO 3: Construyendo SVM con parámetros por defecto...")
print("=" * 60)

# Crear pipeline con TfidfVectorizer y SVM
pipeline_base = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 3),  # Usar unigramas, bigramas y trigramas
        min_df=2,
        max_df=0.95
    )),
    ('svm', SVC(random_state=42))
])

print("\nEntrenando el modelo base...")
pipeline_base.fit(X_train, y_train)

# Predicciones
y_pred_base = pipeline_base.predict(X_test)

# Evaluación
print("\n" + "=" * 60)
print("RESULTADOS DEL MODELO BASE")
print("=" * 60)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred_base):.4f}")
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred_base))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_base, target_names=['No Spam', 'Spam']))

# ============================================
# PASO 4: OPTIMIZAR EL MODELO CON GRID SEARCH
# ============================================
print("\n" + "=" * 60)
print("PASO 4: Optimizando hiperparámetros con Grid Search...")
print("=" * 60)

# Definir el pipeline
pipeline_optimized = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(random_state=42))
])

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    'tfidf__max_features': [2000, 3000, 5000],
    'tfidf__ngram_range': [(1, 2), (1, 3)],
    'tfidf__min_df': [2, 3],
    'tfidf__max_df': [0.9, 0.95],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto']
}

print("\nEjecutando Grid Search (esto puede tomar varios minutos)...")
print(f"Número total de combinaciones: {np.prod([len(v) for v in param_grid.values()])}")

# Realizar Grid Search con validación cruzada
grid_search = GridSearchCV(
    pipeline_optimized,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\n" + "=" * 60)
print("RESULTADOS DE LA OPTIMIZACIÓN")
print("=" * 60)
print(f"\nMejor score en validación cruzada: {grid_search.best_score_:.4f}")
print(f"\nMejores parámetros encontrados:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Evaluar el modelo optimizado
best_model = grid_search.best_estimator_
y_pred_optimized = best_model.predict(X_test)

print("\n" + "=" * 60)
print("RESULTADOS DEL MODELO OPTIMIZADO")
print("=" * 60)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred_optimized):.4f}")
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred_optimized))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_optimized, target_names=['No Spam', 'Spam']))

# Comparación de modelos
print("\n" + "=" * 60)
print("COMPARACIÓN DE MODELOS")
print("=" * 60)
print(f"Accuracy Modelo Base: {accuracy_score(y_test, y_pred_base):.4f}")
print(f"Accuracy Modelo Optimizado: {accuracy_score(y_test, y_pred_optimized):.4f}")
print(f"Mejora: {(accuracy_score(y_test, y_pred_optimized) - accuracy_score(y_test, y_pred_base)):.4f}")

# ============================================
# PASO 5: GUARDAR EL MODELO
# ============================================
print("\n" + "=" * 60)
print("PASO 5: Guardando el modelo...")
print("=" * 60)

# Guardar el mejor modelo
with open('spam_url_detector_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("\n✓ Modelo guardado como 'spam_url_detector_model.pkl'")

# Guardar también la función de preprocesamiento
with open('url_preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocess_url, f)
print("✓ Función de preprocesamiento guardada como 'url_preprocessor.pkl'")

# ============================================
# FUNCIÓN PARA PREDECIR NUEVAS URLs
# ============================================
print("\n" + "=" * 60)
print("EJEMPLOS DE PREDICCIÓN")
print("=" * 60)

def predict_spam(url, model=best_model):
    """Predice si una URL es spam o no"""
    url_processed = preprocess_url(url)
    prediction = model.predict([url_processed])[0]
    probability = model.decision_function([url_processed])[0]
    
    return {
        'url': url,
        'is_spam': bool(prediction),
        'confidence': abs(probability)
    }

# Probar con algunas URLs de ejemplo
test_urls = [
    "https://www.google.com/search",
    "http://free-money-now.suspicious-site.com/click-here",
    "https://github.com/usuario/proyecto",
    "http://buy-now-limited-offer.xyz/deal",
    "https://www.wikipedia.org/wiki/Machine_Learning"
]

print("\nProbando el modelo con URLs de ejemplo:\n")
for url in test_urls:
    result = predict_spam(url)
    spam_label = "SPAM ⚠️" if result['is_spam'] else "LEGÍTIMA ✓"
    print(f"{spam_label} | Confianza: {result['confidence']:.2f} | {result['url']}")

print("\n" + "=" * 60)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 60)