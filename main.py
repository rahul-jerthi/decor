import streamlit as st
import json
import os
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load product data
with open("img.json", "r") as f:
    products = json.load(f)

# Precompute product features
product_features = []
product_ids = []

for product in products:
    image_path = os.path.join("products", os.path.basename(product["image_path"]))
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model.encode_image(image).cpu().numpy()
        product_features.append(feature)
        product_ids.append(product["id"])
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")

product_features = np.array([f[0] for f in product_features])  # Convert to NumPy

# --- Streamlit UI ---
st.set_page_config(page_title="Image Recommendation System", layout="wide")
st.title("🖼️ Product Recommendation using CLIP")

uploaded_file = st.file_uploader("Upload an image to find similar products", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    user_image = Image.open(uploaded_file)
    st.image(user_image, caption="Uploaded Image", width=300)

    # Encode uploaded image
    user_tensor = preprocess(user_image).unsqueeze(0).to(device)
    with torch.no_grad():
        user_feature = model.encode_image(user_tensor).cpu().numpy()

    # Compute cosine similarity
    similarities = cosine_similarity(user_feature, product_features)[0]
    top_indices = similarities.argsort()[-5:][::-1]  # top 5 matches

    st.subheader("🔍 Top Recommendations")
    cols = st.columns(5)  # Display 5 images in a row

    for i, idx in enumerate(top_indices):
        product_id = product_ids[idx]
        matched_product = next((p for p in products if p["id"] == product_id), None)
        if matched_product:
            with cols[i]:
                product_img_path = os.path.join("products", os.path.basename(matched_product["image_path"]))
                st.image(product_img_path, caption=matched_product["name"], use_container_width=True)
                st.markdown(f"**Category:** {matched_product['category']}")
                st.markdown(f"**Tags:** {', '.join(matched_product['tags'])}")
