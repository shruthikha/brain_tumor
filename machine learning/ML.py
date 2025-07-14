import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define dataset paths
train_folder_path = r"C:\Users\shrut\Downloads\archive (1)\Training"
test_folder_path = r"C:\Users\shrut\Downloads\archive (1)\Testing"

# Image size for VGG16
input_size = (224, 224)

# Data augmentation and preprocessing
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(train_folder_path, target_size=input_size, batch_size=128, class_mode='categorical')
test_generator = datagen.flow_from_directory(test_folder_path, target_size=input_size, batch_size=128, class_mode='categorical', shuffle=False)

# Load pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
base_model.trainable = False

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Create new model
model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save the trained model
model.save('netTransfer.h5')

# Load the trained model
model = tf.keras.models.load_model('netTransfer.h5')

# Make predictions on test set
y_pred = np.argmax(model.predict(test_generator), axis=1)
y_true = test_generator.classes

# Compute confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)

# Display confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=train_generator.class_indices.keys(), yticklabels=train_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Compute classification report
report = classification_report(y_true, y_pred, target_names=train_generator.class_indices.keys())
print(report)
