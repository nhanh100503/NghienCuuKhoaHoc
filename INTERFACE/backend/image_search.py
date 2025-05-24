import numpy as np
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from PIL import Image
import faiss
import joblib
from sklearn.decomposition import PCA
import os
import base64
# Model paths
MODEL_PATHS = {
    'vgg16': 'E:/similarity_image/models/vgg16/vgg16_raw.keras',
    'vgg16_aug': 'E:/similarity_image/models/vgg16/vgg16_aug2_best_params.keras',
    'convnext_v2_aug': 'E:/similarity_image/models/convnext_v2/convnext_v2_best_params_aug2_final',
}


INDEX_DIR = {
    'vgg16': os.getenv('INDEX_VGG16'),
    'convnext_v2': os.getenv('INDEX_CONVNEXT_V2'),
    'alexnet': os.getenv('INDEX_ALEXNET'),
}

RAW_DATASET = os.getenv('RAW_DATASET')


# Directory for FAISS indices and PCA
# INDEX_DIR = r'E:\similarity_image\extract_features'
IMG_FOLDER = os.getenv('RAW_DATASET')

# Get class names from folder names
class_names = [folder for folder in os.listdir(IMG_FOLDER) if os.path.isdir(os.path.join(IMG_FOLDER, folder))]

# PCA configuration
PCA_COMPONENTS = 256
pca = None

# FAISS indices for each class and model
dimension = PCA_COMPONENTS
faiss_indices = {model_type: {class_name: None for class_name in class_names} for model_type in MODEL_PATHS.keys()}

# Image preprocessing for ConvNeXt V2
preprocess_torch = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_preprocess_torch(resize_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def preprocess_keras_image(img):
    """Preprocess image for Keras model."""
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_spoc_features(img, model, model_type):
    """Extract SPoC features from an image."""
    if model_type.startswith('vgg16'):
        spoc_extractor = Model(inputs=model.input, outputs=model.get_layer('block5_pool').output)
        img_array = preprocess_keras_image(img)
        features = spoc_extractor.predict(img_array)
        pooled = np.sum(features, axis=(1, 2))
        normalized = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return normalized[0]
    # elif model_type.startswith('convnext_v2'):
    #     feature_extractor = nn.Sequential(
    #         model.stem,
    #         model.stages  # Up to stage 3
    #     )
    #     preprocess = get_preprocess_torch()  # Chuẩn input AlexNet
    #     img_tensor = preprocess(img).unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         feature_map = feature_extractor(img_tensor)  # [batch, C, H, W]
    #         features = torch.sum(feature_map, dim=[2, 3])  # Sum-pooling
    #         normalized = features / torch.norm(features, dim=1, keepdim=True)
    #     return normalized.cpu().numpy()[0]
    elif model_type.startswith('convnext_v2'):
        preprocess = get_preprocess_torch()  # Chuẩn input (AlexNet-like)
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            x = model.stem(img_tensor)
            for stage in model.stages:
                x = stage(x)  # Đi qua từng stage của ConvNeXt
            x = torch.sum(x, dim=[2, 3])  # Sum-pooling (SPoC)
            x = x / torch.norm(x, dim=1, keepdim=True)  # Normalize
        return x.cpu().numpy()[0]
    elif model_type.startswith('alexnet'):
        preprocess = get_preprocess_torch((227, 227))  # Chuẩn input AlexNet
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            x = model.features(img_tensor)  # (1, 256, 6, 6)
            x = x.view(x.size(0), -1)       # flatten to (1, 256*6*6 = 9216)
            x = x / torch.norm(x, dim=1, keepdim=True)  # normalize
        return x.cpu().numpy()[0]

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def classify_image(img, model, model_type):
    """Classify an image."""
    if model_type.startswith('vgg16'):
        img_array = preprocess_keras_image(img)
        preds = model.predict(img_array)
        class_id = np.argmax(preds[0])
        confidence = np.max(preds[0])
        return class_names[class_id], confidence, preds[0]
    elif model_type.startswith(('convnext_v2', 'alexnet')):
        preprocess = get_preprocess_torch()  # Chuẩn input AlexNet
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(img_tensor)
            preds = torch.softmax(preds, dim=1)
        class_id = torch.argmax(preds[0]).item()
        confidence = preds[0][class_id].item()
        return class_names[class_id], confidence, preds[0].cpu().numpy()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def load_pca(model_type):
    """Load PCA model."""
    global pca
    print(model_type + '99')
    pca_path = os.path.join(f'{INDEX_DIR.get(model_type)}/pca_{model_type}_aug.pkl')
    print(pca_path)
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
    else:
        raise ValueError(f"PCA model not found for {model_type}")

def apply_pca(features):
    """Apply PCA transformation."""
    if pca is None:
        raise ValueError("PCA model not loaded")
    return pca.transform(features.reshape(1, -1))[0]

def load_faiss_index(model_type, class_name):
    """Load FAISS index for a class."""
    index_path = os.path.join( f'{INDEX_DIR.get(model_type)}/{model_type}_aug_class_{class_name}.index')
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    return faiss.IndexFlatL2(dimension)

def save_faiss_index(index, model_type, class_name):
    """Save FAISS index."""
    faiss.write_index(index, os.path.join(f'{INDEX_DIR.get(model_type)}/{model_type}_aug_class_{class_name}.index'))

def load_image_ids(model_type, class_name):
    """Load image IDs for a class."""
    ids_path = os.path.join( f'{INDEX_DIR.get(model_type)}/{model_type}_aug_image_ids_{class_name}.pkl')
    if os.path.exists(ids_path):
        return joblib.load(ids_path)
    return []

def save_image_ids(image_ids, model_type, class_name):
    """Save image IDs."""
    joblib.dump(image_ids, os.path.join( f'{INDEX_DIR.get(model_type)}/{model_type}_aug_image_ids_{class_name}.pkl'))

def search_similar_images(img, model, model_type, threshold, cursor, top_k=200):
    """Search for similar images in the predicted class."""
    pred_class, confidence, preds = classify_image(img, model, model_type)
    print(pred_class)
    load_pca(model_type)
    features = extract_spoc_features(img, model, model_type)
    query_vector = apply_pca(features)
    index = load_faiss_index(model_type, pred_class)
    image_ids = load_image_ids(model_type, pred_class)
    # D, I = index.search(query_vector.astype(np.float32).reshape(1, -1), index.ntotal)
    D, I = index.search(query_vector.astype(np.float32).reshape(1, -1), min(index.ntotal, top_k))
    similar_items = []
    valid_image_ids = []
    similar_scores = []

    for dist, idx in zip(D[0], I[0]):
        if idx < len(image_ids):
            sim_score = 1 - dist
            if sim_score >= threshold:
                valid_image_ids.append(image_ids[idx])
                similar_scores.append(sim_score)

    if valid_image_ids:
        placeholders = ",".join(["%s"] * len(valid_image_ids))
        query = f"SELECT * FROM research WHERE image_id IN ({placeholders})"
        cursor.execute(query, valid_image_ids)
        results = cursor.fetchall()

        # Tạo dict để tra cứu theo image_id
        image_id_to_research = {row['image_id']: row for row in results}

        similar_items = [
            (score, image_id_to_research[img_id])
            for score, img_id in zip(similar_scores, valid_image_ids)
            if img_id in image_id_to_research
        ]
    else:
        similar_items = []

    similar_items.sort(reverse=True, key=lambda x: x[0])
    print(similar_items)

    similar_results = []
    for sim, research in similar_items:
        # Lấy đường dẫn file ảnh từ trường image_field_name
        image_path = os.path.join(RAW_DATASET, pred_class, research['image_field_name'])
        
        # Đọc file ảnh, mã hóa base64
        try:
            with open(image_path, "rb") as img_f:
                image_bytes = img_f.read()
                image_data = base64.b64encode(image_bytes).decode('utf-8')
                image_data = f"data:image/png;base64,{image_data}"
        except Exception as e:
            image_data = ""  # hoặc None, nếu ảnh không tồn tại hoặc lỗi
        
        similar_results.append({
            'similarity': float(f'{sim*100}'),
            'title': research['title'],
            'doi': research['doi'],
            'caption': research['caption'],
            'image_field_name': research['image_field_name'],
            'authors': research['authors'],
            'language': research['language'],
            'image_data': image_data,
        })
    return {
        'predicted_class': pred_class,
        'confidence':  float(f'{confidence*100}'),
        'all_classes': [
            {'label': class_names[i], 'confidence': float(f'{conf * 100:.2f}')}
            for i, conf in enumerate(preds)
        ],
        'similar_images': similar_results,
        # 'total_similar_images': len(similar_results)
    }

# Device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')