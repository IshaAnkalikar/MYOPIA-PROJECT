import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load and preprocess images
def load_and_preprocess_image(image_path, input_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size))
    image = preprocess_input(image)
    return image

# Load images and labels
def load_data(image_dir):
    dataset = []
    labels = []
    label_map = {'No': 2, 'hypermetropia': 0, 'myopia': 1}
    for label_name in label_map.keys():
        image_paths = os.listdir(os.path.join(image_dir, label_name))
        for image_name in image_paths:
            if image_name.endswith('.png'):
                image_path = os.path.join(image_dir, label_name, image_name)
                image = load_and_preprocess_image(image_path, INPUT_SIZE)
                dataset.append(image)
                labels.append(label_map[label_name])
    return np.array(dataset), np.array(labels)

# Load data
image_directory = 'dataset/'
INPUT_SIZE = 224
dataset, labels = load_data(image_directory)

# Split data into train, validation, and test sets
train_images, test_images, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

# Load pre-trained DenseNet121 model
base_model = DenseNet121(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), include_top=False, weights='imagenet')

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
global_average_layer = layers.GlobalAveragePooling2D()
dropout = layers.Dropout(0.5)
output_layer = layers.Dense(3, activation='softmax')

# Create the model
model = models.Sequential([
  base_model,
  global_average_layer,
  dropout,
  output_layer
])

# Compile the model
model.compile(optimizer=optimizers.Adam(lr=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(train_datagen.flow(train_images, train_labels, batch_size=32),
                    steps_per_epoch=len(train_images) / 32,
                    epochs=30,
                    validation_data=(val_images, val_labels),
                    callbacks=[early_stop])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Save the model
model.save('myopia_and_hypermetropia_prediction_model.h5')

# Predict class probabilities
y_pred_probs = model.predict(test_images)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion Matrix
cm = confusion_matrix(test_labels, y_pred)
class_names = ['hypermetropia', 'myopia', 'No']

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(test_labels, y_pred, target_names=class_names))
