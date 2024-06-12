import cv2
import pytesseract
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Path to the Tesseract executable (update this path if needed)
pytesseract.pytesseract.tesseract_cmd = r'D:\Programs\tesseract.exe'

# Load the image
img = cv2.imread("sample_image3.png")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# Apply adaptive thresholding to improve text segmentation
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Find contours (potential text regions)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define and train CNN model
# Replace this with your CNN model architecture and training process
# For demonstration, let's use a simple CNN architecture using TensorFlow/Keras
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img.shape[0], img.shape[1], 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Assuming you have a labeled dataset with positive and negative samples
# Extract features for each ROI and prepare corresponding labels
# For demonstration, let's use random features and labels
X_train = np.random.rand(len(contours), img.shape[0], img.shape[1], 1)  # Dummy features (replace with actual feature extraction)
y_train = np.random.choice([0, 1], size=len(contours))  # Dummy labels (replace with actual labels)

# Train CNN model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Prepare test data for evaluation
# Assuming you have a separate test set with features and labels
# For demonstration, let's use the same random features and labels
X_test = np.random.rand(len(contours), img.shape[0], img.shape[1], 1)  # Dummy features for test set
y_test = np.random.choice([0, 1], size=len(contours))  # Dummy labels for test set

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_binary = np.round(y_pred)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Iterate through contours and classify text regions using CNN
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Crop the region of interest (ROI) containing potential text
    roi = img[y:y+h, x:x+w]

    # Assuming your CNN model expects input of shape (height, width, channels)
    # You may need to preprocess the ROI before feeding it to the model
    # For example, resizing, normalization, etc.

    # Make prediction using CNN model
    prediction = model.predict(np.expand_dims(roi, axis=0))
    prediction_binary = np.round(prediction)

    # If CNN predicts positive for text, apply OCR
    if prediction_binary == 1:
        text = pytesseract.image_to_string(roi)
        print("Extracted text:", text)
