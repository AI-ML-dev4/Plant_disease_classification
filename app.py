import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from PIL import Image
from tqdm import tqdm

# Load saved model
model = joblib.load("models/plant_pathology_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# feature extraction function 
def extract_features(images):
    feature_list = []
    for gray in tqdm(images):
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        # grascale to rgb for color histogram
        rgb = cv2.cvtColor((gray * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)

        # HOG features
        hog_feat = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)

        # Color histogram 
        hist = cv2.calcHist([rgb], [0, 1, 2], None, [8, 8, 8], [0,256, 0,256, 0,256])
        color_feat = cv2.normalize(hist, hist).flatten()

        # Texture features 
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        texture_feat = hist_lbp.astype("float")
        texture_feat /= (texture_feat.sum() + 1e-6)

        # Edge features
        edges = cv2.Canny((gray * 255).astype('uint8'), 100, 200)
        edge_feat = edges.flatten()[:1000]  

        combined = np.concatenate([hog_feat, color_feat, texture_feat, edge_feat])
        feature_list.append(combined)
    
    return feature_list

# streamlit
st.title("Plant Disease Classifier 🌿")
st.write("Upload a leaf image of apple trees to classify its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # resize input image
    image = image.resize((128, 128))

    features = extract_features([np.array(image)])
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Disease: **{predicted_label}**")
