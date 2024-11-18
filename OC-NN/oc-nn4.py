import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Cargar los datos de MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalización de las imágenes
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

# Normalizar entre 0 y 1
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Construcción del Autoencoder Profundo
input_layer = Input(shape=(784,))
encoded = Dense(512, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)

# Decodificador
decoded = Dense(256, activation='relu')(encoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')

# Entrenar el autoencoder solo con las imágenes normales (dígitos 0)
x_train_normal = x_train[y_train == 0]
autoencoder.fit(x_train_normal, x_train_normal, epochs=25, batch_size=128)

# Extraer el codificador (encoder) del autoencoder entrenado
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense_2').output)

# Obtener las características codificadas (representaciones de las imágenes) para el conjunto de test
encoded_x_train = encoder.predict(x_train_normal)
encoded_x_test = encoder.predict(x_test)

# Crear la red neuronal Feed-Forward utilizando las características extraídas por el encoder
model = Sequential([
    Dense(64, activation='relu', input_dim=encoded_x_train.shape[1]),  # Entrada: tamaño de la representación codificada
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Salida: 1 para normal, 0 para anómalo
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Etiquetas: Las imágenes normales tendrán la etiqueta 1 y las anómalas la etiqueta 0 (dígitos 1-9)
y_train_normal = np.ones(len(encoded_x_train))  # Todas las imágenes normales son positivas

# Filtrar imágenes anómalas: Digitos 1-9
x_train_anomalous = x_train[y_train != 0]
encoded_x_anomalous = encoder.predict(x_train_anomalous)

# Etiquetas de anomalías: 0
y_train_anomalous = np.zeros(len(encoded_x_anomalous))

# Unir las imágenes normales y anómalas
x_train_combined = np.concatenate([encoded_x_train, encoded_x_anomalous], axis=0)
y_train_combined = np.concatenate([y_train_normal, y_train_anomalous], axis=0)

# Entrenar la red neuronal con las características codificadas
model.fit(x_train_combined, y_train_combined, epochs=50, batch_size=128)

# Predicción para imágenes de test
encoded_x_test = encoder.predict(x_test)
y_pred_nn = model.predict(encoded_x_test)

# Visualización de resultados
def show_images(images, labels, n=10):
    plt.figure(figsize=(10, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        ax.imshow(images[i].reshape(28, 28), cmap="gray")
        ax.set_title("Anomalous" if labels[i] < 0.5 else "Normal")
        ax.axis("off")
    plt.show()

# Mostrar imágenes clasificadas como normales o anómalas
show_images(x_test, y_pred_nn, n=10)

# Mostrar estadísticas
normal_images = np.sum(y_pred_nn >= 0.5)
anomalous_images = np.sum(y_pred_nn < 0.5)
print(f"Normal Images: {normal_images}")
print(f"Anomalous Images: {anomalous_images}")
