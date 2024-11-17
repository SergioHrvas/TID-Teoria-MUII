import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


# Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalización de las imágenes
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Dividir en conjuntos normales y anómalos (suponiendo que 0 es normal)
x_train_normal = x_train[y_train == 0]
y_train_normal = y_train[y_train == 0]

x_test_normal = x_test[y_test == 0]
y_test_normal = y_test[y_test == 0]

x_test_anomalous = x_test[y_test != 0]
y_test_anomalous = y_test[y_test != 0]


# Verificar la distribución de clases en el conjunto de datos de validación
unique_classes, counts = np.unique(y_test, return_counts=True)
plt.bar(unique_classes, counts)
plt.xlabel("Clase")
plt.ylabel("Cantidad de imágenes")
plt.title("Distribución de clases en el conjunto de validación")
plt.show()

# Definir el modelo con más capas
model = keras.Sequential([
 keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='sigmoid', kernel_initializer='he_normal', 
                       kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(128, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.1),
    #keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    #keras.layers.Dropout(0.2),  # Dropout con una tasa de 20%
    #keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
    #keras.layers.Dropout(0.2),  # Dropout con una tasa de 20%
    #keras.layers.Dense(16, activation='sigmoid', kernel_initializer='he_normal'),
    #keras.layers.Dropout(0.2),  # Dropout con una tasa de 20%
    keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal')
])

# Compilar el modelo
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00002), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento solo con las imágenes normales
history = model.fit(x_train_normal, np.ones(len(x_train_normal)), epochs=100, batch_size=32, verbose=0, validation_data=(x_test_normal, np.ones(len(x_test_normal))))

# Calcular los puntajes de decisión para todo el conjunto de prueba
decision_scores = model.predict(x_test)

# Ajustar el umbral para clasificar las imágenes como normales o anómalas
threshold = 0.9 # Ajusta este valor según el comportamiento de tu modelo

normal_anomalies = [1 if score > threshold else 0 for score in decision_scores]

# Función para mostrar imágenes clasificadas como normales o anómalas
def show_samples(images, titles, n=10):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.show()

# Ejemplos de imágenes detectadas como normales
show_samples(x_test_normal[:10], ["Normal" if not x else "Anomalía" for x in normal_anomalies[:10]])

# Ejemplos de imágenes detectadas como anómalas
show_samples(x_test_anomalous[:10], ["Anomalía" if x else "Normal" for x in normal_anomalies[:10]])

# Graficar la distribución de los puntajes de decisión
plt.hist(decision_scores[y_test == 0], bins=50, alpha=0.5, label="Normales")
plt.hist(decision_scores[y_test != 0], bins=50, alpha=0.5, label="Anómalas")
plt.axvline(x=threshold, color='r', linestyle='--', label='Umbral')
plt.legend()
plt.title("Distribución de los puntajes de decisión")
plt.show()

# Mostrar resultados
print(f"Tasa de detección en datos normales (debería ser baja): {np.mean(normal_anomalies)}")

# Graficar la pérdida de entrenamiento
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')  # Si tienes datos de validación
plt.title('Curva de Aprendizaje - Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Graficar la precisión de entrenamiento
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')  # Si tienes datos de validación
plt.title('Curva de Aprendizaje - Precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()