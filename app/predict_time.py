import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import xception
from tensorflow.keras.models import load_model, Model
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import gc

# Ensure that TensorFlow does not allocate all GPU memory (optional, if using GPU)
# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     try:
#         tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     except:
#         pass

# Load the saved model
model = load_model("../model/2024-11-20/model.keras")

# Reconstruct the MultiLabelBinarizer
# Load the labels DataFrame to fit the binarizer
labels_dir = "../data/analog_clocks/label.csv"  # Update the path if necessary
labels_df = pd.read_csv(labels_dir)

# Transform labels to multi-label binary format
labels_df["tuples"] = [
    ("h" + str(x), "m" + str(y)) for x, y in zip(labels_df["hour"], labels_df["minute"])
]
binarizer = MultiLabelBinarizer()
y = binarizer.fit_transform(labels_df["tuples"])

# Prepare the Xception model for feature extraction
prebuilt_model = xception.Xception(include_top=True, weights="imagenet")
xception_model = Model(
    inputs=prebuilt_model.input, outputs=prebuilt_model.layers[-2].output
)


def predict_from_saved_model(
    image_paths, model, xception_model, binarizer, image_size=(299, 299), plot=False
):
    """
    Predict the time displayed on analog clock images using the saved model.

    Parameters:
    - image_paths: List of paths to the images.
    - model: The trained Keras model loaded from 'model.keras'.
    - xception_model: Xception model for feature extraction.
    - binarizer: MultiLabelBinarizer fitted on the original labels.
    - image_size: Target size for image resizing.
    - plot: Whether to display the images with predictions.

    Returns:
    - predicted_times: List of predicted times as tuples (hour, minute).
    """
    images_list = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=image_size)
        img_arr = image.img_to_array(img)
        images_list.append(img_arr)

    images_array = np.array(images_list)
    images_preprocessed = xception.preprocess_input(images_array)

    # Extract features using Xception model
    features = xception_model.predict(images_preprocessed)

    # Use the saved model to predict
    predictions = model.predict(features)

    # Map predictions back to labels
    num_hours = 12  # Assuming hours are labeled from 'h0' to 'h11'
    num_minutes = 12  # Assuming minutes are labeled from 'm0' to 'm11'

    hour_predictions = predictions[:, :num_hours]
    minute_predictions = predictions[:, num_hours:]

    hour_max = np.argmax(hour_predictions, axis=1)
    minute_max = np.argmax(minute_predictions, axis=1)

    predicted_times = []
    for h_idx, m_idx in zip(hour_max, minute_max):
        h_label = binarizer.classes_[h_idx]
        m_label = binarizer.classes_[
            num_hours + m_idx
        ]  # Offset by num_hours for minute labels
        # Extract numerical values from labels
        predicted_hour = int(h_label[1:])
        predicted_minute = int(m_label[1:])
        predicted_times.append((predicted_hour, predicted_minute))

    if plot:
        # Plot the images with predicted times
        plt.figure(figsize=(15, 5))
        for i, img_path in enumerate(image_paths):
            img = image.load_img(img_path)
            plt.subplot(1, len(image_paths), i + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(
                f"Predicted: {predicted_times[i][0]:02d}:{predicted_times[i][1]:02d}"
            )
        plt.show()

    return predicted_times


# Example usage:
# Provide the list of image paths you want to predict
image_dir = "../data/analog_clocks/samples"  # Update the path if necessary
image_files = os.listdir(image_dir)
image_paths = [os.path.join(image_dir, fname) for fname in image_files]

# Call the predict function
predicted_times = predict_from_saved_model(
    image_paths, model, xception_model, binarizer, plot=True
)

# Print the predictions
for img_path, pred_time in zip(image_paths, predicted_times):
    print(
        f"Image: {os.path.basename(img_path)}, Predicted Time: {pred_time[0]:02d}:{pred_time[1]:02d}"
    )
