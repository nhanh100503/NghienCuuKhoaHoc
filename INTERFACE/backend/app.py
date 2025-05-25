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
from tensorflow.keras.models import load_model, Model


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  
print("Max content length:", app.config['MAX_CONTENT_LENGTH'])

CORS(app)  

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"error": "File too large!"}), 413

IMAGE_SIZE = (224, 224)
RAW_DATASET_DIR = os.getenv('RAW_DATASET')

MODEL_FILES = {
    'MobileNetV2': os.getenv('MODEL_MOBILENETV2'),
    'ResNet101': os.getenv('MODEL_RESNET101'),
    'EfficientNetB0': os.getenv('MODEL_EFFICIENTNETB0'),
    'InceptionV3': os.getenv('MODEL_INCEPTIONV3'),
    'InceptionResNetV2': os.getenv('MODEL_INCEPTIONRESNETV2'),
    'InceptionV4': os.getenv('MODEL_INCEPTIONV4'),

}


MODEL_OTHERS_FILES = {
    'convnext_v2': os.getenv('MODEL_CONVNEXT_V2'),
    'alexnet': os.getenv('MODEL_ALEXNET'),
    'vgg16': os.getenv('MODEL_VGG16'),
}


CLASS_INDICES_DIR = os.getenv('CLASS_INDICES_DIR')
IMAGE_ROOT_DIR = os.getenv('IMAGE_ROOT_DIR')


def connect_db():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE")
    )

def preprocess_pytorch_image(img, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)
    # img_tensor = transform(img).unsqueeze(0).to(device)  # đưa lên device luôn

    return img_tensor

# Preprocessing for Keras Models (VGG16)
def preprocess_keras_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

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
            torch.nn.Dropout(0.3),
            torch.nn.Linear(in_feats, 1024, bias=False),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 512, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 11),
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval().to(device)
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
        transforms.Resize(IMAGE_SIZE),
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
        confidence = probs[0, class_idx].item()  # xác suất tương ứng

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
            transforms.Resize(IMAGE_SIZE),
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



    

# Tính độ tương đồng
def compute_similarity(input_image_path, model_name, threshold):
    if model_name not in model_loader.models:
        return "Model không hợp lệ", [], 0

    full_model = model_loader.models[model_name]
    extractor = model_loader.extractors[model_name]
    print(model_name + "2")
   
    if(model_name == "InceptionV3" or model_name == "InceptionResNetV2" or model_name == "InceptionV4"):
        print(model_name + "1")
        image_array = preprocess_image(input_image_path, size=(299, 299))
    else:
        image_array = preprocess_image(input_image_path, size=(224, 224))

    if(model_name == "InceptionV4"):
        predicted_class, confidence = classify_image_pytorch(full_model, input_image_path)
    else:
        predicted_class, confidence = classify_image(full_model, image_array)

    print(f"Phân lớp ảnh đầu vào: {predicted_class}")

    if(model_name == "InceptionV3" or model_name == "InceptionResNetV2" or model_name == "InceptionV4"):
        input_feature = extract_feature(input_image_path, extractor, model_name=model_name, size=(299, 299))
        print(f"Kích thước đặc trưng ảnh đầu vào: {input_feature.shape}")
    else:
        input_feature = extract_feature(input_image_path, extractor, model_name=model_name, size=(224, 224))
        print(f"Kích thước đặc trưng ảnh đầu vào: {input_feature.shape}")

    # Kết nối cơ sở dữ liệu
    conn = connect_db()
    cursor = conn.cursor()
    if(model_name == "InceptionV3" ):
        model_name = "InceptionV3_TC"
    elif(model_name == "InceptionResNetV2"):
        model_name = "INCEPTIONRESNETV2_TC"
    elif(model_name == "InceptionV4"):
        model_name = "InceptionV4_TC"

    # Truy vấn tất cả các đặc trưng từ bảng feature với model_name được chọn
    cursor.execute("""
        SELECT f.image_id, f.feature_vector, r.image_field_name, r.doi, r.title, r.caption, r.authors
        FROM feature f
        JOIN research r ON f.image_id = r.image_id
        WHERE f.model_name = %s AND r.class_name = %s
    """, (model_name, predicted_class))
    rows = cursor.fetchall()
    print(f"Số lượng ảnh trong cơ sở dữ liệu cho {model_name}: {len(rows)}")

    # Tính độ tương đồng cosine
    similar_images = []
    for image_id, feature_str, image_name, doi, title, caption, authors in rows:
        try:
            # Giải mã bytes thành chuỗi và chuyển thành mảng số
            if isinstance(feature_str, bytes):
                feature_str = feature_str.decode('utf-8')
            db_vector = np.array(list(map(float, feature_str.split(','))))
            print(f"Kích thước đặc trưng từ DB cho {image_name}: {db_vector.shape}")
            similarity = cosine_similarity([input_feature], [db_vector])[0][0]
            print(f"Độ tương đồng với {image_name}: {similarity}")
            if similarity >= threshold:
                similar_images.append({
                    'image_id': image_id,
                    'image_field_name': image_name,
                    'similarity': round(float(similarity), 4),
                    'doi': doi,
                    'title': title,
                    'caption': caption,
                    'authors': authors,
                    'similarity': round(float(similarity) * 100, 2),  # chuyển thành %
                })
        except Exception as e:
            print(f"Lỗi với ảnh {image_name}: {e}")
            continue

    similar_images.sort(key=lambda x: x['similarity'], reverse=True)
    total_similar_images = len(similar_images)  # Tổng số ảnh tương đồng

    cursor.close()
    conn.close()

    return predicted_class, similar_images, total_similar_images, confidence



def classify_and_find_similar_pth(b64_image, model_type, threshold):
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

# API endpoint
@app.route('/similarity', methods=['POST'])
def get_similarity():
    if 'image' not in request.files:
        return jsonify({'error': 'Thiếu tệp ảnh'}), 400

    image_file = request.files['image']
    model_name = request.form.get('model_name', 'ResNet101')
    threshold = float(request.form.get('threshold', 0.90))

    # Kiểm tra model_name bắt buộc và hợp lệ
    if not model_name or model_name not in MODEL_FILES:
        return jsonify({'error': 'model_name là bắt buộc và phải hợp lệ (MobileNetV2, ResNet101 hoặc EfficientNetB0)'}), 400

    # Lưu ảnh tạm thời để xử lý
    input_image_path = f"temp_{image_file.filename}"
    image_file.save(input_image_path)

    predicted_class, similar_images, total_similar_images = compute_similarity(input_image_path, model_name, threshold)

    # Xóa ảnh tạm sau khi xử lý
    os.remove(input_image_path)

    response = {
        'predicted_class': predicted_class,
        'similar_images': similar_images,
        'total_similar_images': total_similar_images  # Thêm tổng số ảnh tương đồng vào response
    }
    return jsonify(response)

TEMP_DIR = r"INTERFACE\backend\temp"


@app.route('/similarity-image', methods=['POST'])
def get_similarity_from_base64_list():
    data = request.get_json()
    model_name = data.get('model_name')
    threshold = data.get('threshold')
    base64_data = data.get('images', {}) 


    if not model_name or (model_name not in MODEL_FILES and model_name not in MODEL_OTHERS_FILES):
        return jsonify({'error': 'model_name không hợp lệ'}), 400


    class_counter = {}
    
    if(model_name == "vgg16" or model_name == "alexnet" or model_name == "convnext_v2"):
       results = classify_and_find_similar_pth(base64_data, model_name, threshold)
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
    app.run(debug=True, host='0.0.0.0', port=5003)