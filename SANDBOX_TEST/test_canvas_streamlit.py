# pip install streamlit-drawable-canvas
from streamlit_drawable_canvas import st_canvas
import streamlit as st

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np

st.title("Streamlit Drawable Canvas Demo")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fill color for objects
    stroke_width=2,
    stroke_color="#000000",             # Color of the drawing brush
    background_color="#eee",             # Background color of the canvas
    height=400,
    width=600,
    drawing_mode="freedraw",            # Set drawing mode
    key="canvas_example",
)

# Display the drawn image data
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)

# Display the JSON data of drawn objects
if canvas_result.json_data is not None:
    st.write("Drawn Objects JSON:")
    st.json(canvas_result.json_data)

