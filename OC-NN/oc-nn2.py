import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU

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

# Paso 1: Definir el Autoencoder y la Red Neuronal para una sola clase (One-Class)

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


# Red neuronal para detectar anomalías
anomaly_layer = Dense(256, activation='relu')(decoded)
anomaly_layer = Dropout(0.4)(anomaly_layer)
anomaly_layer = Dense(128, activation='relu')(anomaly_layer)
anomaly_layer = Dropout(0.4)(anomaly_layer)
anomaly_layer = Dense(64, activation='relu')(anomaly_layer)
anomaly_layer = Dropout(0.4)(anomaly_layer)
anomaly_layer = Dense(32, activation='relu')(anomaly_layer)
anomaly_layer = Dropout(0.4)(anomaly_layer)
anomaly_layer = Dense(16, activation='relu')(anomaly_layer)
output = Dense(1, activation='linear')(anomaly_layer)
oc_nn = Model(inputs=input_img, outputs=output)

# Paso 2: Métrica personalizada para calcular el accuracy basado en hinge loss
def binary_accuracy_hinge(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(tf.sign(y_pred), y_true), tf.float32))
    
# Compilación del modelo
optimizer = Adam(learning_rate=0.000005)  # Aumentar el learning rate para mejor convergencia

oc_nn.compile(
    optimizer=optimizer, 
    loss='hinge', 
    metrics=[binary_accuracy_hinge]  # Métrica personalizada
)

# Callback para detener el entrenamiento si no mejora
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    mode='min', 
    restore_best_weights=True
)

# Paso 3: Preparación de las etiquetas y datos
y_train_normal_labels = np.ones(len(x_train_normal)) * -1  # Etiquetas de normales (-1)

# Paso 4: Entrenamiento del modelo
history = oc_nn.fit(
    x_train_normal, 
    y_train_normal_labels, 
    epochs=75, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[early_stop]
)
# Paso 3: Evaluación en el conjunto de prueba

# Solo seleccionar imágenes de dígitos 0 para el conjunto de prueba (normales)
x_test_normal = x_test[y_test == 0]
y_test_normal = np.ones(len(x_test_normal)) * -1  # Etiqueta -1 para normales

# Solo seleccionar imágenes de dígitos diferentes a 0 para el conjunto de prueba (anómalas)
x_test_anomalous = x_test[y_test != 0]
y_test_anomalous = np.ones(len(x_test_anomalous))  # Etiquetas 1 para anómalas

# Combinar datos de prueba (normales y anómalos)
x_test_combined = np.concatenate((x_test_normal, x_test_anomalous), axis=0)
y_test_combined = np.concatenate((y_test_normal, y_test_anomalous), axis=0)

# Predecir las puntuaciones de salida para todo el conjunto de prueba
y_pred_combined = oc_nn.predict(x_test_combined).flatten()

# Filtrar las puntuaciones de los datos normales
normal_scores = y_pred_combined[y_test_combined == -1]  # Solo los datos normales (etiquetados como -1)

# Calcular el percentil para obtener el valor r (umbral)
r = np.percentile(normal_scores, 5)  # 98% percentil de los datos normales

# Calcular las puntuaciones de decisión (distancia al centro de normalidad)
decision_scores = y_pred_combined - r  # La diferencia con el centro idealizado

# Clasificar según las puntuaciones de decisión
predictions = np.where(decision_scores >= 0, -1, 1)  # -1: Normal, 1: Anómalo


# Mostrar el número de imágenes normales y anómalas clasificadas
normal_images = np.sum(predictions == -1)  # Imágenes normales clasificadas
anomalous_images = np.sum(predictions == 1)  # Imágenes anómalas clasificadas



# Paso 5: Visualización del entrenamiento
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Curva de entrenamiento y validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Paso 6: Evaluación del modelo
# Combinar normales y anómalos para el test
x_test_combined = np.concatenate([x_test_normal, x_test_anomalous], axis=0)
y_test_combined = np.concatenate([y_test_normal, y_test_anomalous], axis=0)

# Evaluar en el conjunto de prueba
loss, accuracy = oc_nn.evaluate(x_test_combined, y_test_combined)
print(f"Pérdida en prueba: {loss:.4f}")
print(f"Precisión en prueba: {accuracy:.4f}")

# Mostrar el histograma de las puntuaciones de decisión
plt.hist(decision_scores, bins=50, alpha=0.5, label='Test Data')
plt.axvline(x=r, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.show()

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

# Identificar los índices de los ceros predichos como anómalos (incorrectos)
incorrect_indices = (y_test == 0) & (predictions == 1)  # Ceros que el modelo clasificó como anómalos

# Identificar los índices de los ceros correctamente predichos como normales
correct_indices = (y_test == 0) & (predictions == -1)  # Ceros que el modelo clasificó correctamente como normales

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
correct_indices = (y_test == 0) & (predictions == -1)  # Ceros clasificados como normales

# Filtrar las imágenes correctamente clasificadas
correctly_classified_images = x_test[correct_indices]

# Mostrar el número total de ceros correctamente clasificados como normales
print(f"Número de ceros correctamente clasificados como normales: {len(correctly_classified_images)}")

# Visualizar las primeras 20 imágenes correctamente clasificadas
show_correctly_classified(correctly_classified_images, n=20, cols=5)