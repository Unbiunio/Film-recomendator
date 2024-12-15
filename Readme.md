## Preprocesamiento


En el archivo src/data_processing.py:

Limpieza de datos:
Eliminar duplicados.
Manejar valores nulos (rellenando o eliminando, según corresponda).
Feature engineering:
Normaliza/escalar las calificaciones (rating) si es necesario.
Codificar variables categóricas como theme usando técnicas como One-Hot Encoding.
Agrupar usuarios/películas según calificaciones promedio para obtener características adicionales, como popularidad.

### Explicación del código:
Cargar el dataset: La función load_data lee el archivo CSV y lanza un error si no existe.
Limpieza de datos:
Elimina duplicados.
Maneja valores nulos rellenando la columna rating con la mediana y eliminando filas con datos críticos faltantes.
Codificación de theme:
Usa One-Hot Encoding para transformar la columna categórica theme en variables binarias.
Normalización de rating:
Escala los valores de rating entre 0 y 1 para que los modelos funcionen mejor.
Guardar datos procesados: Exporta el dataset limpio a un nuevo archivo CSV.

## Selección de modelo

En el archivo src/model_training.py:

Eligir un enfoque de recomendación:
Filtrado colaborativo:
Basado en usuarios (user-based collaborative filtering) o en ítems (item-based).
Podemos usar bibliotecas como Surprise o implicit.

Filtrado basado en contenido:

Utilizar similitudes entre películas usando características como theme, production_year, etc.

Método híbrido:

Combinar ambos enfoques para mejorar la precisión.

Entrena el modelo:

Divide los datos en entrenamiento y validación.
Ajusta los hiperparámetros para optimizar el rendimiento del modelo.

### Explicación del código:

Cargar del dataset limpio:
Usar load_cleaned_data para leer el archivo CSV preprocesado.
Preparación de datos para Surprise:
Convertir los datos en un formato compatible con la librería Surprise usando la clase Reader. Esto incluye definir el rango de valores de las calificaciones (en este caso, entre 0 y 1).

Entrenamiento del modelo:
Divide los datos en conjunto de entrenamiento (80%) y prueba (20%) usando train_test_split.
Entrena un modelo SVD (Singular Value Decomposition), una técnica común en filtrado colaborativo.
Guardar el modelo:
Serializa el modelo entrenado con pickle y lo guarda en el archivo model.pkl.

## Generación de predicciones

En el archivo src/model_prediction.py:

Cargar el modelo entrenado (models/model.pkl).
Generar predicciones personalizadas para cada usuario:
Encontrar las 3 películas más relevantes para cada usuario basándonos en las puntuaciones predichas.
Guardar las predicciones en formato JSON en predictions/predictions.json.

Ejemplo de estructura:

json
Copiar código
{
    "target": {
        "user1": ["Movie1", "Movie2", "Movie3"],
        "user2": ["Movie4", "Movie5", "Movie6"]
    }
}

### Explicación del código:
Cargar el modelo:

Usar la función load_model para deserializar el modelo entrenado desde model.pkl.
Generar recomendaciones:

Para cada usuario en el dataset:
Predecir la calificación para todas las películas que no haya calificado.
Ordenar las películas por calificación predicha en orden descendente.
Seleccionar las top 3 películas.
Almacenar las recomendaciones en un diccionario con la estructura:
json
Copiar código
{
    "user1": ["Movie1", "Movie2", "Movie3"],
    "user2": ["Movie4", "Movie5", "Movie6"]
}
Guardar las recomendaciones:

Serializar el diccionario de recomendaciones en formato JSON en el archivo predictions/predictions.json.

## Validación del modelo

Métrica de evaluación: Calcula el k-precision (precisión en el top 3 recomendaciones) para validar el rendimiento de tu sistema.

Realizar pruebas con distintos enfoques (colaborativo, basado en contenido o híbrido) para seleccionar el mejor.


## Documentación de entrega

Explicación de la lógica detrás del enfoque.
Incluir pasos para ejecutar el proyecto.
Asegúrarse de listar correctamente las dependencias en requirements.txt.
Revisar que los archivos cumplan con la estructura del repositorio.
Verificar que el archivo predictions.json tenga el formato correcto.

## Tecnologías

Procesamiento de datos: pandas, numpy
Recomendadores: scikit-learn, Surprise, implicit
Evaluación: scikit-learn.metrics
Serialización del modelo: pickle

