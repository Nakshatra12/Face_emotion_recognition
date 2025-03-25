import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Clear previous TensorFlow session to prevent variable conflicts
K.clear_session()


# Define image dimensions and parameters
IMG_WIDTH, IMG_HEIGHT = 48, 48
BATCH_SIZE = 32
EPOCHS = 40


# Ensure images are processed correctly
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)


# Define dataset paths
train_path = r"D:\face_emotion_detection\archive (2)\images\images\train"
val_path = r"D:\face_emotion_detection\archive (2)\images\images\validation"


# Load training data
train_generator = train_datagen.flow_from_directory(
    train_path,  # Corrected path
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    color_mode="rgb",  # Ensures 3 channels instead of 1
    class_mode='categorical'
)

# Load validation data
val_generator = val_datagen.flow_from_directory(
    val_path,  # Corrected path
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode='categorical'
)


# Define the model
def create_model():
    base_model = tf.keras.applications.VGG16(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)  # Ensure it matches RGB input
    )
    

import tensorflow as tf
from tensorflow.keras.applications import VGG16  # Change as per your model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

# Load Pretrained Model (Change to the model you're using, e.g., ResNet50, MobileNetV2, etc.)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# Freeze the base model layers
base_model.trainable = False  # Now, this won't give an error



# Add custom layers on top of the base model
x = Flatten()(base_model.output)  # Flatten the output from base model
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(7, activation='softmax')(x)  # Connect output layer properly


# Define the final model
model = Model(inputs=base_model.input, outputs=output_layer)  # ✅ Now correctly linked

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Early stopping to avoid overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# Save the model
model.save("emotion_detection_model.h5")


# Display training summary
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()


# Real-time Face Emotion Detection using Laptop Camera
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load trained emotion detection model
model = load_model('emotion_detection_model.h5')  

# Define image dimensions (should match your model's input size)
IMG_WIDTH, IMG_HEIGHT = 48, 48  # Adjust based on dataset

# Emotion labels (adjust based on training dataset)
emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is captured

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (IMG_WIDTH, IMG_HEIGHT))
        face_resized = face_resized / 255.0  # Normalize
        face_resized = np.expand_dims(face_resized, axis=0)  # Expand dimensions

        # Predict emotion
        predictions = model.predict(face_resized)
        emotion_index = np.argmax(predictions)  # Get highest probability index
        emotion_label = emotion_labels.get(emotion_index, "Unknown")  # Convert index to label

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow('Face Emotion Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()