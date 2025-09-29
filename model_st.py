import os
import pickle
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import torch
import clip

# ---------- Config ----------
PRODUCTS_JSON = "img.json"         
PRODUCTS_DIR = "products"          
FEATURES_PKL = "product_features.pkl"
TOP_K = 5

# ---------- Caching helpers ----------
@st.cache_resource(show_spinner=False)
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device

@st.cache_data(ttl=24*3600, show_spinner=False)  # cache product features for a day
def load_or_build_features(force_rebuild=False):
    
    # Try to load saved file
    if not force_rebuild and os.path.exists(FEATURES_PKL):
        try:
            with open(FEATURES_PKL, "rb") as f:
                saved = pickle.load(f)
            # Ensure features become a (N,512) float array
            raw = saved.get("product_features")
            product_features = np.vstack([np.array(x).flatten() for x in raw])
            return {
                "product_features": product_features,
                "product_ids": saved.get("product_ids"),
                "products": saved.get("products")
            }
        except Exception as e:
            st.warning(f"Failed to load {FEATURES_PKL}: {e}. Will rebuild features.")
    
    # If we're here, build features from images
    model, preprocess, device = load_clip_model()
    import json
    with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
        products = json.load(f)

    product_ids = []
    features_list = []

    for p in products:
        product_ids.append(p["id"])
        image_path = os.path.join(PRODUCTS_DIR, os.path.basename(p["image_path"]))
        try:
            img = Image.open(image_path).convert("RGB")
            img_t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(img_t).cpu().numpy().flatten()
            features_list.append(feat)
        except Exception as e:
            st.warning(f"Could not process image {image_path}: {e}")
            # append zeros so indexing doesn't break (or skip — here we append zeros)
            features_list.append(np.zeros(512, dtype=float))
    
    product_features = np.vstack(features_list).astype(float)
    # Save stable representation
    with open(FEATURES_PKL, "wb") as f:
        pickle.dump({
            "product_features": product_features,   # (N,512) np array
            "product_ids": product_ids,
            "products": products
        }, f)
    return {"product_features": product_features, "product_ids": product_ids, "products": products}

# ---------- UI ----------
st.set_page_config(page_title="CLIP Product Recommender", layout="wide")
st.title("🖼️ Wall Image → Product Recommender (CLIP)")
st.markdown("Upload a photo (e.g. `wall.jpg`) and get visually similar product recommendations.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Settings")
    rebuild = st.checkbox("Force rebuild product features (slow)", value=False)
    top_k = st.number_input("How many recommendations (top K)?", min_value=1, max_value=12, value=TOP_K)
    st.write("Features file:", FEATURES_PKL)
    if st.button("Rebuild & Save Features"):
        # clear cache and rebuild
        load_or_build_features.clear()
        data = load_or_build_features(force_rebuild=True)
        st.success(f"Rebuilt and saved features. Products: {len(data['product_ids'])}")

with col2:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg","jpeg","png"])
    st.caption("Tip: use a close-up photo of the wall or scene you want matching products for.")

# Load features (may be cached)
with st.spinner("Loading product features..."):
    data = load_or_build_features(force_rebuild=rebuild)
product_features = data["product_features"]   # (N,512)
product_ids = data["product_ids"]
products = data["products"]

# Validate shapes
if product_features.ndim != 2 or product_features.shape[1] != 512:
    st.error(f"product_features shape is {product_features.shape} — expected (N, 512). Try rebuilding features.")
else:
    if uploaded_file is not None:
        # Display uploaded image
        try:
            user_img = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Cannot open uploaded image: {e}")
            st.stop()
        st.image(user_img, caption="Uploaded Image", use_column_width=False, width=320)

        # Encode uploaded image
        model, preprocess, device = load_clip_model()
        user_tensor = preprocess(user_img).unsqueeze(0).to(device)
        with torch.no_grad():
            user_feat = model.encode_image(user_tensor).cpu().numpy().flatten().reshape(1, -1)

        # Compute similarities
        # Ensure product_features is float64 or float32 array
        pf = np.array(product_features, dtype=float)
        sims = cosine_similarity(user_feat, pf)[0]   # shape (N,)
        top_indices = sims.argsort()[-top_k:][::-1]

        st.subheader(f"Top {top_k} recommendations")
        # Prepare display grid: 3 columns per row (adjustable)
        cols_per_row = 3
        rows = (top_k + cols_per_row - 1) // cols_per_row
        idx = 0
        for r in range(rows):
            cols = st.columns(cols_per_row)
            for c in cols:
                if idx >= top_k:
                    break
                prod_idx = top_indices[idx]
                prod_id = product_ids[prod_idx]
                matched = next((p for p in products if p["id"] == prod_id), None)
                if matched is None:
                    c.write("No product metadata")
                else:
                    # load product image (file path might be relative or remote)
                    prod_path = os.path.join(PRODUCTS_DIR, os.path.basename(matched["image_path"]))
                    # If file missing, try the original path
                    if not os.path.exists(prod_path) and os.path.exists(matched["image_path"]):
                        prod_path = matched["image_path"]

                    try:
                        c.image(prod_path, caption=matched.get("name", "product"), use_column_width=True)
                    except Exception:
                        c.write("Image not found")
                    c.markdown(f"**{matched.get('name','-')}**")
                    c.markdown(f"Category: {matched.get('category','-')}")
                    tags = matched.get("tags", [])
                    if isinstance(tags, list):
                        tags = ", ".join(tags)
                    c.markdown(f"Tags: {tags}")
                    c.markdown(f"Score: {sims[prod_idx]:.4f}")
                idx += 1

        # Optional: show raw similarity values table
        if st.expander("Show raw similarity scores"):
            import pandas as pd
            top_data = []
            for i in top_indices:
                m = next((p for p in products if p["id"] == product_ids[i]), {})
                top_data.append({
                    "id": product_ids[i],
                    "name": m.get("name","-"),
                    "score": float(sims[i])
                })
            st.dataframe(pd.DataFrame(top_data))

    else:
        st.info("Upload an image to get recommendations.")
