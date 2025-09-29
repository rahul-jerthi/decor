import os
import json
import shutil
import uuid
from PIL import Image
import streamlit as st

PRODUCTS_JSON = "img.json"
PRODUCTS_DIR = "products"

# ---------- Helpers ----------
def load_products():
    if os.path.exists(PRODUCTS_JSON):
        with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_products(products):
    with open(PRODUCTS_JSON, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=4, ensure_ascii=False)

# ---------- UI ----------
st.title("🛠 Product Manager")

products = load_products()

# --- Show product list with delete buttons ---
st.subheader("Current Products")
if products:
    for idx, p in enumerate(products):
        cols = st.columns([2, 3, 1])
        img_path = os.path.join(PRODUCTS_DIR, os.path.basename(p["image_path"]))
        if os.path.exists(img_path):
            cols[0].image(img_path, width=80)
        cols[1].write(f"**{p['name']}**\nID: {p['id']}\nCategory: {p.get('category','')}")
        if cols[2].button("Delete", key=f"del_{idx}"):
            # Delete image file
            if os.path.exists(img_path):
                os.remove(img_path)
            # Remove from list
            products.pop(idx)
            save_products(products)
            st.rerun()
else:
    st.info("No products yet.")

# --- Add new product ---
st.subheader("Add New Product")
with st.form("add_product_form", clear_on_submit=True):
    name = st.text_input("Product Name")
    category = st.text_input("Category")
    tags = st.text_input("Tags (comma separated)")
    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Add Product")
    if submitted:
        if not name or not category or not image_file:
            st.error("Name, category, and image are required.")
        else:
            os.makedirs(PRODUCTS_DIR, exist_ok=True)
            # Unique ID
            prod_id = str(uuid.uuid4())
            # Save image
            ext = os.path.splitext(image_file.name)[1]
            img_filename = f"{prod_id}{ext}"
            img_path = os.path.join(PRODUCTS_DIR, img_filename)
            Image.open(image_file).convert("RGB").save(img_path)
            # Add to products list
            new_product = {
                "id": prod_id,
                "name": name,
                "category": category,
                "tags": [t.strip() for t in tags.split(",") if t.strip()],
                "image_path": img_path
            }
            products.append(new_product)
            save_products(products)
            st.success(f"Product '{name}' added.")
            st.rerun()

# --- Trigger feature rebuild ---
if st.button("Recompute Features Now"):
    pass
    # from main import load_or_build_features
    # load_or_build_features.clear()
    # load_or_build_features(force_rebuild=True)
    # st.success("Features recomputed and saved.")
