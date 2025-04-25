import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# TensorFlow Model Prediction
def model_prediction(uploaded_image):
    model = tf.keras.models.load_model('C:/Users/ROMEZ/Desktop/Plant Disease Datasets/Plant_Disease_Dataset/trained_model.keras')
    
    # Process the uploaded image
    image = Image.open(uploaded_image)
    image = image.resize((128, 128))  # Resize to match model input size
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    
    # Predict
    detection = model.predict(input_arr)
    result_index = np.argmax(detection)
    return result_index

# Sidebar with Styling
st.sidebar.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Plant Disease Detector"])

# Common Style for Headers
header_style = """
<style>
h1, h2 {
    color: #2e7d32;
    font-family: 'Arial', sans-serif;
}
</style>
"""
st.markdown(header_style, unsafe_allow_html=True)

# Home Page
if app_mode == "Home":
    st.markdown("<h1>🌿 Plant Disease Detection System</h1>", unsafe_allow_html=True)
    st.image("C:/Users/ROMEZ/Desktop/Plant Disease Datasets/Plant_Disease_Dataset/home_page.jpg", use_container_width=True)
    st.markdown("""
    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px;">
    <h2>Welcome to the Plant Disease Detection System! 🌿🔍</h2>
    <p>Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases.</p>
    <ul>
        <li><strong>Accuracy:</strong> State-of-the-art machine learning techniques.</li>
        <li><strong>User-Friendly:</strong> Simple and intuitive interface.</li>
        <li><strong>Fast:</strong> Receive results in seconds.</li>
    </ul>
    <p><strong>Get Started:</strong> Click on the "Plant Disease Detector" page in the sidebar!</p>
    </div>
    """, unsafe_allow_html=True)

# About Page
elif app_mode == "About":
    st.markdown("<h1>About the Project</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px;">
    <p>This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on GitHub.</p>
    <h3>Dataset Content:</h3>
    <ul>
        <li>Train: 70,295 images</li>
        <li>Validation: 17,572 images</li>
        <li>Test: 33 images</li>
    
    </li>
    <p> This project is developed for educational purposes and was undertaken by Jp Romez. It is part of the requirements for the subject of Methods of Research. Throughout the development of this project, extensive research and analysis were conducted to explore various methodologies and approaches. The aim was to not only fulfill the academic requirements but also to contribute valuable insights and practical knowledge to the field of study. The project showcases a comprehensive understanding of the subject matter and reflects the dedication and hard work put forth by the developer. Special acknowledgments go to the instructors and peers who provided guidance and support throughout the project's duration. </p>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Plant Disease Detector Page
elif app_mode == "Plant Disease Detector":
    st.markdown("<h1>Plant Disease Detector</h1>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", width=200)
    
        # Detect Button with Styling
        detect_button_style = """
        <style>
        div.stButton > button:first-child {
            background-color: #4caf50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 10px;
        }
        div.stButton > button:first-child:hover {
            background-color: #388e3c;
            color: white;
        }
        </style>
        """
        st.markdown(detect_button_style, unsafe_allow_html=True)
        
        if st.button("Detect"):
            st.write("Analyzing the uploaded image...")
            result_index = model_prediction(uploaded_image)
            
            # Define Class Names
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            
            # Display Result
            st.success(f"Disease Detected: {class_name[result_index]}")
