import numpy as np
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Cargar MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Remodelar las imágenes para que sean vectores de 784 elementos (28x28)
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

# Solo seleccionar imágenes de dígitos 0 para entrenar el autoencoder (imágenes normales)
x_train_normal = x_train[y_train == 0]
y_train_normal = y_train[y_train == 0]

# Solo seleccionar imágenes de dígitos diferentes a 0 para probar el modelo (anómalas)
x_test_anomalous = x_test[y_test != 0]
y_test_anomalous = y_test[y_test != 0]

# Paso 1: Definir y entrenar el Autoencoder
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu', name='encoded_layer')(encoded)

decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(decoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer="adam", loss='binary_crossentropy')

# Entrenar el autoencoder solo con imágenes normales (dígitos 0)
autoencoder.fit(x_train_normal, x_train_normal, epochs=10, batch_size=64)

# Paso 2: Usar el encoder para extraer características
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded_layer').output)

encoded_x_train = encoder.predict(x_train_normal)
encoded_x_test = encoder.predict(x_test)

# Paso 3: Crear una Red Neuronal Feed-Forward para clasificación
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=16))  # Capa de entrada correspondiente a las características extraídas
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Salida de 1 neurona para clasificar normal vs anómalo
optimizer2 = Adam(learning_rate=0.000001)

early_stop = EarlyStopping(
    monitor='loss',       # O 'val_loss' si usas validación
    patience=10,           # Detén después de 5 épocas sin mejora
    mode='min',           # Queremos minimizar la pérdida
    restore_best_weights=True  # Restaurar los mejores pesos
)

model.compile(optimizer=optimizer2, loss='binary_crossentropy', metrics=['accuracy'])

# Etiquetas para la clasificación (0 para normales, 1 para anómalas)
y_train_normal_labels = np.zeros(len(y_train_normal))  # Todas las imágenes normales etiquetadas como 0
y_test_anomalous_labels = np.ones(len(x_test_anomalous))  # Imágenes anómalas etiquetadas como 1

# Entrenar la red neuronal con las características del autoencoder
model.fit(encoded_x_train, y_train_normal_labels, epochs=400, batch_size=128, callbacks=[early_stop])

# Paso 4: Hacer predicciones para las imágenes de prueba (todas las imágenes)
encoded_x_test = encoder.predict(x_test)  # Extraer características de las imágenes de prueba
y_pred_nn = model.predict(encoded_x_test)  # Predecir la probabilidad de cada imagen

# Paso 5: Calcular puntuaciones de decisión y clasificar las imágenes
decision_scores = y_pred_nn.flatten()  # Puntuaciones de decisión para cada imagen de prueba
threshold = np.percentile(decision_scores, 25)  # Umbral: percentil 50 para clasificar normal/anómalo
predictions = (decision_scores >= threshold).astype(int)

# Contar cuántas imágenes normales y anómalas fueron correctamente clasificadas
normal_images = np.sum(predictions == 0)  # Normal
anomalous_images = np.sum(predictions == 1)  # Anómalo

print(f"Normal Images: {normal_images}")
print(f"Anomalous Images: {anomalous_images}")


# Función para visualizar las imágenes clasificadas
def show_images(images, predictions, n=40, label='Normal', cols=10):
    # Filtrar las imágenes según la clase (Normal o Anomalous)
    if label == 'Normal':
        filtered_images = images[predictions == 0]
    else:
        filtered_images = images[predictions == 1]
    
    rows = (n // cols) + (n % cols != 0)  # Determinar el número de filas necesario

    # Mostrar las primeras 'n' imágenes filtradas
    plt.figure(figsize=(cols * 2, rows * 2))  # Tamaño ajustado a las filas y columnas
    for i in range(min(n, len(filtered_images))):
        plt.subplot(rows, cols, i + 1)  # Distribución de las imágenes en filas y columnas
        plt.imshow(filtered_images[i].reshape(28, 28), cmap="gray")
        plt.title(f'{label} {i+1}')
        plt.axis('off')
    plt.show()

# Mostrar imágenes normales (predicción 0) y anómalas (predicción 1)
# show_images(x_test, predictions, n=40, label='Normal', cols=10)   # Primeras 20 imágenes normales
# show_images(x_test, predictions, n=40, label='Anomalous', cols=10)

def show_images_by_digit(images, predictions, labels, digit, n=20, cols=5, title_label='Prediction'):
    """
    Muestra imágenes de un dígito específico con su predicción.

    :param images: array de imágenes a mostrar.
    :param predictions: array de predicciones asociadas a las imágenes.
    :param labels: etiquetas reales de las imágenes (0 para normal, 1 para anómalo).
    :param digit: dígito específico a filtrar para mostrar (por ejemplo, 0).
    :param n: número de imágenes a mostrar.
    :param cols: número de columnas en el grid.
    :param title_label: texto a mostrar en las etiquetas de las predicciones.
    """
    # Filtrar imágenes y predicciones según el dígito real
    digit_indices = (labels == digit)
    filtered_images = images[digit_indices]
    filtered_predictions = predictions[digit_indices]
    filtered_labels = labels[digit_indices]
    
    rows = (n // cols) + (n % cols != 0)  # Determinar número de filas necesario
    plt.figure(figsize=(cols * 2, rows * 2))  # Ajustar tamaño del grid

    for i in range(min(n, len(filtered_images))):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(filtered_images[i].reshape(28, 28), cmap="gray")
        pred_label = 'Normal' if filtered_predictions[i] == 0 else 'Anomalous'
        true_label = 'Normal' if filtered_labels[i] == 0 else 'Anomalous'
        plt.title(f'{title_label}: {pred_label}\nTrue: {true_label}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()



# Mostrar imágenes normales y anómalas con predicciones
# show_images_by_digit(x_test, predictions, y_test, digit=0, n=20, cols=5, title_label='Prediction')
# show_images_by_digit(x_test, predictions, y_test, digit=1, n=20, cols=5, title_label='Prediction')


# Mostrar las imágenes mal clasificadas
def show_misclassified(images, n=20, cols=5, title="Ceros como Anómalos"):
    rows = (n // cols) + (n % cols != 0)
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(min(n, len(images))):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap="gray")
        plt.title(f'{title} {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Identificar los índices de los ceros predichos como anómalos
incorrect_indices = (y_test == 0) & (predictions == 1)

# Filtrar las imágenes mal clasificadas
misclassified_images = x_test[incorrect_indices]
misclassified_predictions = predictions[incorrect_indices]
misclassified_labels = y_test[incorrect_indices]

# Mostrar el número total de ceros clasificados erróneamente como anómalos
print(f"Número de ceros mal clasificados como anómalos: {len(misclassified_images)}")

# Visualizar las primeras 20 imágenes mal clasificadas
show_misclassified(misclassified_images, n=20, cols=5)


# Mostrar las imágenes correctamente clasificadas
def show_correctly_classified(images, n=20, cols=5, title="Ceros como Normales"):
    rows = (n // cols) + (n % cols != 0)
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(min(n, len(images))):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap="gray")
        plt.title(f'{title} {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()



# Identificar los índices de los ceros correctamente predichos como normales
correct_indices = (y_test == 0) & (predictions == 0)

# Filtrar las imágenes correctamente clasificadas
correctly_classified_images = x_test[correct_indices]

# Mostrar el número total de ceros correctamente clasificados como normales
print(f"Número de ceros correctamente clasificados como normales: {len(correctly_classified_images)}")

# Visualizar las primeras 20 imágenes correctamente clasificadas
show_correctly_classified(correctly_classified_images, n=20, cols=5)