import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Cargar datos de MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Seleccionar solo las imágenes del dígito "0" como la clase normal
x_train = x_train[y_train == 0]
x_test_normal = x_test[y_test == 0]
x_test_anomalous = x_test[y_test != 0]

# Escalar las imágenes para tener valores de píxeles entre 0 y 1
x_train = x_train.astype('float32') / 255.
x_test_normal = x_test_normal.astype('float32') / 255.
x_test_anomalous = x_test_anomalous.astype('float32') / 255.

# Expandir las dimensiones para que las imágenes sean de la forma (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test_normal = np.expand_dims(x_test_normal, -1)
x_test_anomalous = np.expand_dims(x_test_anomalous, -1)

# Definir el modelo de autoencoder
def build_autoencoder():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

autoencoder = build_autoencoder()
autoencoder.summary()

# Entrenar el modelo de autoencoder
autoencoder.fit(x_train, x_train, epochs=25, batch_size=32, validation_split=0.2,  verbose=0)

# Calcular el error de reconstrucción en el conjunto de pruebas "normal" (dígitos 0)
reconstructions = autoencoder.predict(x_test_normal)
mse = np.mean(np.power(x_test_normal - reconstructions, 2), axis=(1, 2, 3))
threshold = np.percentile(mse, 80)  # Definir umbral al percentil 90
print(f"Umbral de anomalía: {threshold}")

# Evaluar en datos normales
reconstructions_normal = autoencoder.predict(x_test_normal)
mse_normal = np.mean(np.power(x_test_normal - reconstructions_normal, 2), axis=(1, 2, 3))
normal_anomalies = mse_normal > threshold

# Evaluar en datos anómalos
reconstructions_anomalous = autoencoder.predict(x_test_anomalous)
mse_anomalous = np.mean(np.power(x_test_anomalous - reconstructions_anomalous, 2), axis=(1, 2, 3))
anomalous_detected = mse_anomalous > threshold

# Mostrar resultados
print(f"Tasa de detección en datos normales (debería ser baja): {np.mean(normal_anomalies)}")
print(f"Tasa de detección en datos anómalos (debería ser alta): {np.mean(anomalous_detected)}")

def show_samples(images, titles, n=15):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.show()

# Ejemplos de imágenes detectadas como normales
show_samples(x_test_normal[:15], ["Normal" if not x else "Anomalía" for x in normal_anomalies[:15]])

# Ejemplos de imágenes detectadas como anómalas
show_samples(x_test_anomalous[:15], ["Anomalía" if x else "Normal" for x in anomalous_detected[:15]])
def show_samples(images, titles, n=15):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.show()

# Ejemplos de imágenes detectadas como normales
show_samples(x_test_normal[:15], ["Normal" if not x else "Anomalía" for x in normal_anomalies[:15]])

# Ejemplos de imágenes detectadas como anómalas
show_samples(x_test_anomalous[:15], ["Anomalía" if x else "Normal" for x in anomalous_detected[:15]])
