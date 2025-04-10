import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3, VGG16, EfficientNetB0
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import os

# Set dataset path
project_folder = os.getcwd()
dataset_path = os.path.join(project_folder, "dataset")
print(f"Using dataset from: {dataset_path}")

# Load and Process Dataset
full_df = pd.read_csv(os.path.join(dataset_path, "full_df.csv"))

# Function to create file paths for images
def get_image_path(filename):
    return os.path.join(dataset_path, "ODIR-5K", "Training Images", filename)

# Creating a new DataFrame for individual eye images
data = []
for _, row in full_df.iterrows():
    data.append({"image_path": get_image_path(row["Left-Fundus"]), "labels": row["labels"]})
    data.append({"image_path": get_image_path(row["Right-Fundus"]), "labels": row["labels"]})

image_df = pd.DataFrame(data)
train_df, test_df = train_test_split(image_df, test_size=0.2, random_state=42, stratify=image_df["labels"])

# Data Augmentation
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_dataframe(train_df, x_col="image_path", y_col="labels", target_size=(224, 224), batch_size=32, class_mode="categorical")
test_generator = datagen.flow_from_dataframe(test_df, x_col="image_path", y_col="labels", target_size=(224, 224), batch_size=32, class_mode="categorical", shuffle=False)

# Load Pretrained Models
def create_model(base_model):
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(8, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=x)

inception = create_model(InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
vgg16 = create_model(VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
efficientnet = create_model(EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))

# Custom CNN Model
def custom_cnn():
    model = Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    return model

model_cnn = custom_cnn()

# Compile Models
for model in [model_cnn, inception, vgg16, efficientnet]:
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Models Without Validation
def train_and_save_model(model, name):
    model.fit(train_generator, epochs=10)
    model.save(f"{name}_model.h5")
    print(f"{name} model trained and saved successfully!")

train_and_save_model(model_cnn, "cnn")
train_and_save_model(inception, "inception")
train_and_save_model(vgg16, "vgg16")
train_and_save_model(efficientnet, "efficientnet")

# Evaluate Model on Test Set
def evaluate_model(model, name):
    loss, accuracy = model.evaluate(test_generator)
    print(f"{name} Model Test Accuracy: {accuracy * 100:.2f}%")

evaluate_model(model_cnn, "CNN")
evaluate_model(inception, "InceptionV3")
evaluate_model(vgg16, "VGG16")
evaluate_model(efficientnet, "EfficientNetB0")
