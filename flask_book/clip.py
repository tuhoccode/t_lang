import os
import glob
import base64
import faiss
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from io import BytesIO
import matplotlib.pyplot as plt

app = Flask(__name__)

model = SentenceTransformer('clip-ViT-B-32')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(CURRENT_DIR, 'static', 'images')
OUTPUT_INDEX_PATH = os.path.join(CURRENT_DIR, 'vector.index')

def generate_clip_embeddings(images_path, model, max_images=100):
    image_paths = glob.glob(os.path.join(images_path, '*.webp'))[:max_images]
    
    if not image_paths:
        print(f"Không tìm thấy ảnh .webp trong thư mục: {images_path}")
        return [], []
    
    print(f"Tìm thấy {len(image_paths)} ảnh")
    
    embeddings = []
    valid_image_paths = []
    
    for img_path in image_paths:
        try:
            image = Image.open(img_path)
            image = image.convert("RGB")  
            embedding = model.encode(image)
            embeddings.append(embedding)
            valid_image_paths.append(img_path)
            print(f"Đã xử lý ảnh: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"Không thể xử lý ảnh {img_path}: {e}")
    
    print(f"Tổng số ảnh đã xử lý thành công: {len(embeddings)}")
    return embeddings, valid_image_paths

def create_faiss_index(embeddings, image_paths, output_path):
    if not embeddings or not image_paths:
        print("Không có dữ liệu để tạo index")
        return None
        
    dimension = len(embeddings[0])
    index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(index)
    vectors = np.array(embeddings).astype(np.float32)
    index.add_with_ids(vectors, np.array(range(len(embeddings))))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    faiss.write_index(index, output_path)
    print(f"Đã lưu index tại {output_path}")
    
    with open(output_path + '.paths', 'w') as f:
        for img_path in image_paths:
            f.write(img_path + '\n')
    
    return index

def load_faiss_index(index_path):
    """Load FAISS index and image paths"""
    if not os.path.exists(index_path):
        print(f"Không tìm thấy file index tại {index_path}")
        return None, []
        
    index = faiss.read_index(index_path)
    with open(index_path + '.paths', 'r') as f:
        image_paths = [line.strip() for line in f]
    print(f"Đã tải index từ {index_path}")
    return index, image_paths

def search_similar_images(query, model, index, image_paths, top_k=3):
    if isinstance(query, str):
        query_features = model.encode(query)
    else:
        query_features = model.encode(query)
    
    query_features = query_features.astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query_features, top_k)
    
    similar_images = [image_paths[idx] for idx in indices[0]]
    similarities = distances[0]
    
    return similar_images, similarities

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def main():
    if not os.path.exists(IMAGES_PATH):
        print(f"Thư mục images không tồn tại tại: {IMAGES_PATH}")
        return
    
    if not os.path.exists(OUTPUT_INDEX_PATH):
        print("Đang tạo embeddings và FAISS index...")
        embeddings, image_paths = generate_clip_embeddings(IMAGES_PATH, model, max_images=100)
        if not embeddings:
            print("Không có embeddings để tạo index")
            return
        
        print("Đang tạo FAISS index...")
        index = create_faiss_index(embeddings, image_paths, OUTPUT_INDEX_PATH)
        if index is None:
            print("Không thể tạo index")
            return
        print("Hoàn tất tạo index!")

if __name__ == '__main__':
    main()  
    app.run(debug=True)
