import tensorflow as tf
from tensorflow.keras import layers, regularizers
import numpy as np
import matplotlib.pyplot as plt
import sys
import random

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalización de las imágenes
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Seleccionar solo las imágenes de la clase normal (ejemplo: clase 0)
x_train_normal = x_train[y_train == 0]
x_test_normal = x_test[y_test == 0]
x_test_anomalous = x_test[y_test != 0]

# Remodelar para el modelo
x_train_normal = x_train_normal.reshape(-1, 28 * 28)
x_test_normal = x_test_normal.reshape(-1, 28 * 28)
x_test_anomalous = x_test_anomalous.reshape(-1, 28 * 28)

# Definir el autoencoder
input_dim = x_train_normal.shape[1]

# Construcción del autoencoder
encoder = tf.keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
])

decoder = tf.keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])

autoencoder = tf.keras.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')

# Entrenar el autoencoder solo con datos normales
autoencoder.fit(x_train_normal, x_train_normal, epochs=100, batch_size=64, validation_split=0.2, verbose=0)

# Extraer las características del encoder
x_train_encoded = encoder.predict(x_train_normal)

# Construcción de la red OC-NN (feed-forward) con el encoder preentrenado
oc_nn = tf.keras.Sequential([
    encoder,
    layers.Dense(64, activation='linear', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1, activation='sigmoid')  # Para decidir si es normal (1) o anómalo (0)
])

# Compilar OC-NN
oc_nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005), loss='binary_crossentropy', metrics=['accuracy'])

# Crear etiquetas (1 para normal) para entrenar la OC-NN
y_train_ocnn = np.ones(len(x_train_normal))

# Entrenar OC-NN solo con datos normales
history = oc_nn.fit(x_train_normal, y_train_ocnn, epochs=200, batch_size=64, validation_split=0.1, verbose=0)

# Obtener los puntajes de decisión en el conjunto de prueba
decision_scores_normal = oc_nn.predict(x_test_normal)
decision_scores_anomalous = oc_nn.predict(x_test_anomalous)

# Definir un umbral para la detección de anomalías (ajustado a un valor más bajo)
threshold = np.percentile(decision_scores_normal, 90)

# Los normales deberían tener puntajes de decisión más bajos (por debajo del umbral)
predictions_normal = (decision_scores_normal <= threshold).astype(int)

# Los anómalos deberían tener puntajes de decisión más altos (por encima del umbral)
predictions_anomalous = (decision_scores_anomalous > threshold).astype(int)

def show_samples(data, predictions, title, num_samples=40):
    # Determinar el número de columnas por fila (puedes ajustarlo)
    ncols = 10
    nrows = (num_samples + ncols - 1) // ncols  # Número de filas necesarias

    plt.figure(figsize=(15, 3 * nrows))  # Ajustar el tamaño de la figura para que se vean bien las imágenes

    indices = random.sample(range(len(data)), num_samples)  # Selección aleatoria de imágenes

    for i, idx in enumerate(indices):
        plt.subplot(nrows, ncols, i + 1)  # Organizar en filas y columnas
        plt.imshow(data[idx].reshape(28, 28), cmap="gray")
        plt.title("Normal" if predictions[idx] == 1 else "Anómalo")
        plt.axis("off")

    plt.suptitle(title)
    plt.show()

# Mostrar muestras de datos normales con predicciones
show_samples(x_test_normal, predictions_normal, "Predicciones en datos normales")

# Mostrar muestras de datos anómalos con predicciones
show_samples(x_test_anomalous, predictions_anomalous, "Predicciones en datos anómalos")

# Mostrar distribuciones separadas
plt.hist(decision_scores_normal, bins=50, alpha=0.5, label="Normales", color='blue')
plt.hist(decision_scores_anomalous, bins=50, alpha=0.5, label="Anómalas", color='red')
plt.axvline(threshold, color='green', linestyle='--', label="Umbral")
plt.legend()
plt.title("Distribución de los puntajes de decisión")
plt.xlabel("Puntaje")
plt.ylabel("Frecuencia")
plt.show()

# Mostrar curva de pérdida y precisión de entrenamiento
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Evaluación de resultados
accuracy_normal = np.mean(predictions_normal)  # Porcentaje de normales correctamente clasificados
accuracy_anomalous = np.mean(predictions_anomalous)  # Porcentaje de anómalos correctamente clasificados

print(f"Tasa de detección de normales: {accuracy_normal:.2f}")
print(f"Tasa de detección de anómalos: {accuracy_anomalous:.2f}")
