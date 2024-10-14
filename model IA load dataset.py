#encoding:utf-8
import tensorflow as tf
from tensorflow.keras.layers import Rescaling
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and prepare dataset
img_height = 180
img_width = 180
batch_size = 32

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Prétraitement des données
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.map(lambda x, y: (Rescaling(1./255)(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (Rescaling(1./255)(x), y))

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Création du modèle CNN
num_classes = len(train_dataset.class_names)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entraînement du modèle
epochs = 10
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset
)

# Évaluation du modèle
loss, acc = model.evaluate(validation_dataset)
print("Validation Loss: ", loss)
print("Validation Accuracy: ", acc)

# Sauvegarde du modèle
model.save('modele_classification_vegetal.h5')

# Prédiction avec une nouvelle image 
#A completer par le bon chemin de l'image
img = image.load_img("path_to_new_image.jpg", target_size=(img_height, img_width))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class = train_dataset.class_names[np.argmax(predictions)]
print(f"Predicted class: {predicted_class}")

# Affichage du graphique de l'évolution de la précision et du loss
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')
plt.show()
