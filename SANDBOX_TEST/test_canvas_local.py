from streamlit_drawable_canvas import st_canvas
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt 
import pandas as pd

st.title("Draw → Grayscale → 28x28 Processor")

def pixel_visualize(img):
    # Convert PIL Image → NumPy array if needed
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap="gray")

    width, height = img.shape
    threshold = img.max() / 2.5
    
    for x in range(width):
        for y in range(height):
            ax.annotate(
                str(round(img[x][y], 2)),
                xy=(y, x),
                color="white" if img[x][y] < threshold else "black"
            )

            
# Draw canvas
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
)

# Process the output
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype("uint8"))

    # 1) grayscale
    img = ImageOps.grayscale(img)

    # 2) resize 28x28
    img = img.resize((28, 28))

    # 3) invert so digit is dark on light background
    img = ImageOps.invert(img)

    # 4) to [0,1] float32
    x = np.array(img).astype("float32") / 255.0   # (28, 28)

    # 5) example for Keras
    x_keras = x.reshape(1, 28, 28, 1)

    simple_model_filename = "SANDBOX_TEST\\mnist_cnn_model.keras"

    # 6. Load the saved model
    loaded_model = keras.models.load_model(simple_model_filename)
    print(f"Model loaded from {simple_model_filename}")

    # 7. Predict
    predictions = loaded_model.predict(x_keras)

    # Get predicted class (0–9)
    predicted_digit = predictions.argmax(axis=1)[0]

    # Get probability of the predicted digit
    predicted_prob = predictions[0][predicted_digit]

    st.subheader("Simple Model Prediction")
    st.write(f"**Digit:** {predicted_digit}")
    st.write(f"**Confidence:** {predicted_prob:.4f}")
    
    

    overkill_model_filename = "SANDBOX_TEST\\overkill_mnist_cnn_model.keras"

    # 6. Load the saved model
    overkill_loaded_model = keras.models.load_model(overkill_model_filename)
    print(f"Model loaded from {overkill_model_filename}")

    # 7. Predict
    overkill_predictions = overkill_loaded_model.predict(x_keras)

    # Get predicted class (0–9)
    overkill_predicted_digit = overkill_predictions.argmax(axis=1)[0]

    # Get probability of the predicted digit
    overkill_predicted_prob = overkill_predictions[0][overkill_predicted_digit]

    st.subheader("Overkill Model Prediction")
    st.write(f"**Digit:** {overkill_predicted_digit}")
    st.write(f"**Confidence:** {overkill_predicted_prob:.4f}")


    # st.write(loaded_model.summary())

    st.subheader("Pixel Visualization")
    img_array = np.array(img)
    # This prints the image with value annotations
    pixel_visualize(img_array)
    st.pyplot(plt)



