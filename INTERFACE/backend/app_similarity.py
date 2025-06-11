from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import mysql.connector
from flask_cors import CORS
import base64
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dotenv import load_dotenv
load_dotenv()
import uuid
from PIL import Image
from io import BytesIO
import json
from load_model import *
from image_search import *
from werkzeug.exceptions import RequestEntityTooLarge


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  

CORS(app) 


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"error": "File too large!"}), 413

IMAGE_SIZE = (224, 224)
TEMP_DIR = r"temp"

RAW_DATASET_DIR = os.getenv('RAW_DATASET')
CLASS_INDICES_DIR = os.getenv('CLASS_INDICES_DIR')

MODEL_FILES = {
    'ResNet101': os.getenv('MODEL_RESNET101'),
    'InceptionV4': os.getenv('MODEL_INCEPTIONV4'),

}

MODEL_OTHERS_FILES = {
    'convnext_v2': os.getenv('MODEL_CONVNEXT_V2'),
    'vgg16': os.getenv('MODEL_VGG16'),
}

def connect_db():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),   
        database=os.getenv("MYSQL_DATABASE")
    )

# Load class indices từ thư mục dataset
def load_class_indices():
    datagen = ImageDataGenerator(rescale=1./255)
    dataset = datagen.flow_from_directory(
        CLASS_INDICES_DIR,
        target_size=IMAGE_SIZE,
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )
    return dataset.class_indices

class InceptionV4Extractor(torch.nn.Module):
    def __init__(self, base_model):
        super(InceptionV4Extractor, self).__init__()
        self.base_model = base_model
        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling

    def forward(self, x):
        features = self.base_model.forward_features(x)
        pooled = self.pooling(features)           # shape: (batch, 1536, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # shape: (batch, 1536)
        return pooled

class ModelLoader:
    def __init__(self):
        self.models = {}
        self.extractors = {}
        self.types = {}

        for model_name, model_path in MODEL_FILES.items():

            if not os.path.exists(model_path):
                print(f"Không tìm thấy model: {model_path}")
                continue
            if model_name != 'InceptionV4':
                    full_model = load_model(model_path)
                    extractor = Model(
                        inputs=full_model.input,
                        outputs=full_model.get_layer("global_average_pooling2d").output
                    )
                    self.models[model_name] = full_model
                    self.extractors[model_name] = extractor
            else:
                model = self.load_pytorch_model(model_path)
                extractor = InceptionV4Extractor(model)
                self.models[model_name] = model
                self.extractors[model_name] = extractor 

    def load_pytorch_model(self, path):
        model = timm.create_model('inception_v4', pretrained=False, num_classes=11)
        in_feats = model.last_linear.in_features
        model.last_linear = torch.nn.Sequential(
            torch.nn.Dropout(0.25),

            torch.nn.Linear(in_feats, 256, bias=False),    # hidden layer 1
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.326),

            torch.nn.Linear(256, 128, bias=False),         # hidden layer 2
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.267),

            torch.nn.Linear(128, 11)                        # output layer 
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        return model

model_loader = ModelLoader()
class_indices = load_class_indices()

# Tiền xử lý ảnh
def preprocess_image(img_path, size):
    img = load_img(img_path, target_size=size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Phân loại ảnh
def classify_image(model, image_array):
    prediction = model.predict(image_array, verbose=0)[0]
    class_index = np.argmax(prediction)
    class_name = list(class_indices.keys())[list(class_indices.values()).index(class_index)]
    confidence = prediction[class_index] * 100  # chuyển sang %
    return class_name, confidence

def classify_image_pytorch(model, img_path):
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # chuẩn ImageNet
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)  # output logits
        probs = torch.nn.functional.softmax(output, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, class_idx].item() * 100  # xác suất tương ứng

    idx_to_class = {v: k for k, v in class_indices.items()}
    predicted_class = idx_to_class.get(class_idx, "unknown")
    
    return predicted_class, confidence


# Trích xuất đặc trưng
def extract_feature(img_path, extractor, model_name, size):
    if model_name != 'InceptionV4':
        img = load_img(img_path, target_size=size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        features = extractor.predict(img_array, verbose=0)
        return features.flatten()
    else:
        preprocess = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # chuẩn ImageNet
                                 std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = extractor(input_tensor)
            if isinstance(output, torch.Tensor):
                feature_vector = output.cpu().numpy().flatten()
                return feature_vector
            else:
                raise ValueError("Output của model PyTorch không phải tensor.")

       
# # # Tính độ tương đồng
     
# def compute_similarity(input_image_path, model_name, threshold):
#     if model_name not in model_loader.models:
#         return "Model không hợp lệ", [], 0

#     full_model = model_loader.models[model_name]
#     extractor = model_loader.extractors[model_name]
#     print(model_name)
   
#     if(model_name == "InceptionV3" or model_name == "InceptionResNetV2" or model_name == "InceptionV4"):
#         print(model_name + "1")
#         image_array = preprocess_image(input_image_path, size=(299, 299))
#     else:
#         image_array = preprocess_image(input_image_path, size=(224, 224))

#     if(model_name == "InceptionV4"):
#         predicted_class, confidence = classify_image_pytorch(full_model, input_image_path)
#     else:
#         predicted_class, confidence = classify_image(full_model, image_array)

#     print(f"Phân lớp ảnh đầu vào: {predicted_class}")

#     if(model_name == "InceptionV3" or model_name == "InceptionResNetV2" or model_name == "InceptionV4"):
#         input_feature = extract_feature(input_image_path, extractor, model_name=model_name, size=(299, 299))
#         print(f"Kích thước đặc trưng ảnh đầu vào: {input_feature.shape}")
#     else:
#         input_feature = extract_feature(input_image_path, extractor, model_name=model_name, size=(224, 224))
#         print(f"Kích thước đặc trưng ảnh đầu vào: {input_feature.shape}")
   
#     # Kết nối cơ sở dữ liệu
#     conn = connect_db()     
#     cursor = conn.cursor()
#     if(model_name == "InceptionV3" ):
#         model_name = "InceptionV3_TC"
#     elif(model_name == "InceptionResNetV2"):
#         model_name = "INCEPTIONRESNETV2_TC"
#     elif(model_name == "InceptionV4"):
#         model_name = "InceptionV4_TC"

#     # Truy vấn tất cả các đặc trưng từ bảng feature với model_name được chọn
#     cursor.execute("""
#         SELECT f.image_id, f.feature_vector, r.image_field_name, r.doi, r.title, r.caption, r.authors, r.approved_date, r.page_number
#         FROM feature f
#         JOIN research r ON f.image_id = r.image_id
#         WHERE f.model_name = %s AND r.class_name = %s
#     """, (model_name, predicted_class))
#     rows = cursor.fetchall()
#     print(f"Số lượng ảnh trong cơ sở dữ liệu cho {model_name}: {len(rows)}")

#     # Tính độ tương đồng cosine
#     similar_images = []
#     for image_id, feature_str, image_name, doi, title, caption, authors, approved_date, page_number in rows:
#         try:
#             # Giải mã bytes thành chuỗi và chuyển thành mảng số
#             if isinstance(feature_str, bytes):
#                 feature_str = feature_str.decode('utf-8')
#             db_vector = np.array(list(map(float, feature_str.split(','))))
#             print(f"Kích thước đặc trưng từ DB cho {image_name}: {db_vector.shape}")
#             similarity = cosine_similarity([input_feature], [db_vector])[0][0]
#             print(f"Độ tương đồng với {image_name}: {similarity}")
#             if similarity >= threshold:
#                 similar_images.append({
#                     'image_id': image_id,
#                     'image_field_name': image_name,
#                     'similarity': round(float(similarity), 4),
#                     'doi': doi,
#                     'title': title,
#                     'caption': caption,
#                     'authors': authors,
#                     'accepted_date' : approved_date, 
#                     'similarity': round(float(similarity) * 100, 2),  # chuyển thành %
#                     'page_number': page_number
#                 })
#         except Exception as e:
#             print(f"Lỗi với ảnh {image_name}: {e}")
#             continue

#     similar_images.sort(key=lambda x: x['similarity'], reverse=True)
#     total_similar_images = len(similar_images)  # Tổng số ảnh tương đồng

#     cursor.close()
#     conn.close()

#     return predicted_class, similar_images, total_similar_images, confidence

import faiss
import joblib
import os
import numpy as np

def compute_similarity(input_image_path, model_name, threshold, top_k=10):
    if model_name not in model_loader.models:
        return "Model không hợp lệ", [], 0, 0

    full_model = model_loader.models[model_name]
    extractor = model_loader.extractors[model_name]

    # Xác định kích thước đầu vào
    size = (299, 299) if model_name in ["InceptionV3", "InceptionResNetV2", "InceptionV4"] else (224, 224)
    image_array = preprocess_image(input_image_path, size=size)

    # Phân lớp ảnh
    if model_name == "InceptionV4":
        predicted_class, confidence = classify_image_pytorch(full_model, input_image_path)
    else:
        predicted_class, confidence = classify_image(full_model, image_array)

    print(f"📌 Phân lớp ảnh đầu vào: {predicted_class}, độ tin cậy: {confidence}")

    # Trích xuất đặc trưng
    input_feature = extract_feature(input_image_path, extractor, model_name=model_name, size=size).astype('float32')
    input_feature = np.expand_dims(input_feature, axis=0)
    faiss.normalize_L2(input_feature)  # CHUẨN HÓA để dùng cosine similarity

    print(f"✅ Kích thước vector ảnh đầu vào: {input_feature.shape}")

    # Chuẩn hóa tên model
    if model_name == "InceptionV3":
        model_name_faiss = "InceptionV3_TC"
    elif model_name == "InceptionResNetV2":
        model_name_faiss = "INCEPTIONRESNETV2_TC"
    elif model_name == "InceptionV4":
        model_name_faiss = "InceptionV4_TC"
    else:
        model_name_faiss = model_name

    # Load index + id
    index_path = os.path.join(INDEX_DIR.get(model_name_faiss), f"{model_name_faiss}_class_{predicted_class}.index")
    ids_path = os.path.join(INDEX_DIR.get(model_name_faiss), f"{model_name_faiss}_image_ids_{predicted_class}.pkl")
    print(index_path)
    if not os.path.exists(index_path) or not os.path.exists(ids_path):
        return "Không tìm thấy FAISS index", [], 0, confidence

    index = faiss.read_index(index_path)
    image_ids = joblib.load(ids_path)

    # Tìm top-k ảnh tương đồng
    distances, indices = index.search(input_feature, index.ntotal)

    # Kết nối DB để lấy thông tin ảnh
    conn = connect_db()
    cursor = conn.cursor()

    similar_images = []
    for idx, score in zip(indices[0], distances[0]):
        if idx >= len(image_ids):
            continue

        similarity = float(score)  # <-- dùng trực tiếp luôn nếu dùng IndexFlatIP
    
        if similarity >= threshold:
            image_id = image_ids[idx]
            try:
                cursor.execute("""
                    SELECT r.image_field_name, r.doi, r.title, r.caption, r.authors, r.approved_date
                    FROM research r
                    WHERE r.image_id = %s AND r.class_name = %s
                """, (image_id, predicted_class))
                row = cursor.fetchone()
                if row:
                    image_name, doi, title, caption, authors, approved_date = row
                    similar_images.append({
                        'image_id': image_id,
                        'image_field_name': image_name,
                        'similarity': round(similarity * 100, 2),
                        'doi': doi,
                        'title': title,
                        'caption': caption,
                        'authors': authors,
                        'accepted_date': approved_date,
                    })
            except Exception as e:
                print(f"Lỗi khi truy vấn ảnh {image_id}: {e}")
                continue

    similar_images.sort(key=lambda x: x['similarity'], reverse=True)
    total_similar_images = len(similar_images)

    cursor.close()
    conn.close()

    return predicted_class, similar_images, total_similar_images, confidence
           
         
def classify_and_find_similar_pth(image_list, model_type, threshold): 
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    print("Model path:", MODEL_OTHERS_FILES['convnext_v2'])

    try:
        if model_type == "vgg16":
            model = load_model(MODEL_OTHERS_FILES.get('vgg16'), compile=False)
        elif model_type == "convnext_v2":
            model = load_convnextv2_model(MODEL_OTHERS_FILES['convnext_v2'], 11, model_type="convnext_v2")
        elif model_type == "alexnet":
            model = load_alexnet_model(MODEL_OTHERS_FILES.get('alexnet'), 11)
        else:
            return  ValueError("Invalid model type")
    except Exception as e:
        return jsonify({"error": f"Model load failed: {e}"}, status=500)

    # If image_list is a st ring, decode it
    if isinstance(image_list, str):
        try:
            image_list = json.loads(image_list)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            raise ValueError(f"Invalid image_list format: {e}")
        
    results = []
    for image_obj in image_list:    
        try:
            # Decode image from base64
            b64_image = image_obj.get("base64")
            img_bytes = base64.b64decode(b64_image)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            print(model_type)
            # Perform classification and similarity search
            if model_type in ("vgg16", "convnext_v2", "alexnet" ):  
                result = search_similar_images(img, model, model_type, threshold, cursor)
                result['base64'] = b64_image
                result['model_type'] = model_type
                result['name'] = image_obj.get("name")
                result['caption'] = image_obj.get("caption")
                result['authors'] = image_obj.get('authors')
                result['doi'] = image_obj.get('doi')
                result['accepted_date'] = image_obj.get('approved_date')
                result['model'] = model_type
                result['threshold'] = threshold
                result['title'] = image_obj.get('title')  
                results.append(result)
            else:
                results.append({        
                    'name': '',     
                    'predicted_class': '',
                    'similar_images': '',
                    'total_similar_images': '',
                    'base64': '',
                    'confidence': '',
                    'doi': '',
                    'title': '',
                    'authors': '',
                    'accepted_date': ''
                })
        except Exception as e:
            results.append({"error": str(e)})

    return results


@app.route('/similarity-pdf', methods=['POST'])
def get_similarity_pdf_image():
    data = request.get_json()
    model_name = data.get('model_name')
    threshold = data.get('threshold')
    images_json = data.get('images', [])

    if not model_name or (model_name not in MODEL_FILES and model_name not in MODEL_OTHERS_FILES):
        return jsonify({'error': 'model_name không hợp lệ'}), 400


    results = []
    class_counter = {}
    
    if(model_name == "vgg16" or model_name == "alexnet" or model_name == "convnext_v2"):
       results = classify_and_find_similar_pth(images_json, model_name, threshold)
    else:
        for image_obj in images_json:
            try:
                base64_data = image_obj.get("base64", "")
                image_name = image_obj.get("name", "unknown.png")
                doi = image_obj.get("doi", "")
                title = image_obj.get("title", "")
                authors = image_obj.get("authors", "")
                accepted_date = image_obj.get("approved_date", "")

                if base64_data.startswith("data:image"):
                    base64_data = base64_data.split(",")[1]

                if not os.path.exists(TEMP_DIR):
                    os.makedirs(TEMP_DIR)
                img_bytes = base64.b64decode(base64_data)
                temp_filename = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4().hex}_{image_name}")
                with open(temp_filename, 'wb') as f:
                    f.write(img_bytes)
                print(f"Saved temp image: {temp_filename}")

                predicted_class, similar_images, total, confidence = compute_similarity(temp_filename, model_name, threshold)
                print(f"Predicted class: {predicted_class}")


                results.append({
                    'name': image_name,
                    'predicted_class': predicted_class,
                    'similar_images': similar_images,
                    'total_similar_images': total,
                    'base64': base64_data,
                    'confidence': float(confidence),
                    'doi': doi,   
                    'title': title,
                    'authors': authors,
                    'accepted_date': accepted_date,
                    'model': model_name,
                    'threshold': threshold
                })

                os.remove(temp_filename)  # Xóa file tạm

            except Exception as e:
                results.append({'name': image_name, 'error': str(e)})

    return jsonify({
        'results': results,
        'total': len(results)
    })


    
def classify_and_find_similar_single_image_pth(b64_image, model_type, threshold):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    print("Model path:", MODEL_OTHERS_FILES[model_type])

    try:
        if model_type == "vgg16":
            model = load_model(MODEL_OTHERS_FILES.get('vgg16'), compile=False)
        elif model_type == "convnext_v2":
            model = load_convnextv2_model(MODEL_OTHERS_FILES['convnext_v2'], 11, model_type="convnext_v2")
        elif model_type == "alexnet":
            model = load_alexnet_model(MODEL_OTHERS_FILES.get('alexnet'), 11)
        else:
            return  ValueError("Invalid model type")
    except Exception as e:
        return jsonify({"error": f"Model load failed: {e}"}, status=500)

    try:
        if b64_image.startswith("data:image"):
            b64_image = b64_image.split(",")[1]
        img_bytes = base64.b64decode(b64_image)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        print(model_type + "1")

        if model_type in ("vgg16", "convnext_v2", "alexnet" ):  
            result = search_similar_images(img, model, model_type, threshold, cursor)
            results = {
                **result,  
                'base64': b64_image,
                'model_type': model_type,
                'model': model_type,
                'threshold': threshold,
                'total_similar_images': result.get('total_similar_images')
            }
        else:
            results = {}
    except Exception as e:
        print(e)

    return results


@app.route('/similarity-image', methods=['POST'])
def get_similarity_single_image():
    data = request.get_json()
    model_name = data.get('model_name')
    threshold = data.get('threshold')
    base64_data = data.get('images', {}) 

    if not model_name or (model_name not in MODEL_FILES and model_name not in MODEL_OTHERS_FILES):
        return jsonify({'error': 'model_name không hợp lệ'}), 400
    results = {}
    class_counter = {}
    
    if(model_name == "vgg16" or model_name == "alexnet" or model_name == "convnext_v2"):
       results = classify_and_find_similar_single_image_pth(base64_data, model_name, threshold)
    else:
        try:
            if base64_data.startswith("data:image"):
                base64_data = base64_data.split(",")[1]
            if not os.path.exists(TEMP_DIR):
                os.makedirs(TEMP_DIR)
            img_bytes = base64.b64decode(base64_data)
            temp_filename = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4().hex}")
            with open(temp_filename, 'wb') as f:
                f.write(img_bytes)
            print(f"Saved temp image: {temp_filename}")

            predicted_class, similar_images, total, confidence = compute_similarity(temp_filename, model_name, threshold)
            print(f"Predicted class: {predicted_class}")

            results = {
                'predicted_class': predicted_class,
                'similar_images': similar_images,
                'total_similar_images': total,
                'base64': base64_data,
                'confidence': float(confidence),
                'model': model_name,
                'threshold': threshold
            }

            os.remove(temp_filename)  # Xóa file tạm

        except Exception as e:
            print(e)

    return jsonify({
        'results': results,   
        'total': len(results)
    })

  
@app.route('/dataset/<path:filename>')
def get_image(filename):
    return send_from_directory(RAW_DATASET_DIR, filename)

if __name__ == "__main__":
    # Chạy Flask server
    app.run(debug=True, host='0.0.0.0', port=5001)    