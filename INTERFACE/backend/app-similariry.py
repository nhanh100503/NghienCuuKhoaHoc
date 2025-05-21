from flask import Flask, request, jsonify
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

app = Flask(__name__)
CORS(app, resources={r"/similarity": {"origins": "http://localhost:5173"}})

IMAGE_SIZE = (224, 224)

MODEL_FILES = {
    'MobileNetV2': os.getenv('MODEL_MOBILENETV2'),
    'ResNet101': os.getenv('MODEL_RESNET101'),
    'EfficientNetB0': os.getenv('MODEL_EFFICIENTNETB0')
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

# Load mô hình và feature extractor (quản lý nhiều mô hình)
class ModelLoader:
    def __init__(self):
        self.models = {}
        self.extractors = {}
        for model_name, model_path in MODEL_FILES.items():
            if os.path.exists(model_path):
                full_model = load_model(model_path)
                extractor = Model(
                    inputs=full_model.input,
                    outputs=full_model.get_layer("global_average_pooling2d").output
                )
                self.models[model_name] = full_model
                self.extractors[model_name] = extractor
            else:
                print(f"Không tìm thấy file mô hình: {model_path}")

model_loader = ModelLoader()
class_indices = load_class_indices()

# Tiền xử lý ảnh
def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Phân loại ảnh
def classify_image(model, image_array):
    prediction = model.predict(image_array, verbose=0)
    class_index = np.argmax(prediction[0])
    class_name = list(class_indices.keys())[list(class_indices.values()).index(class_index)]
    return class_name

# Trích xuất đặc trưng
def extract_feature(img_path, extractor):
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = extractor.predict(img_array, verbose=0)
    return features.flatten()

# Tính độ tương đồng
def compute_similarity(input_image_path, model_name, threshold):
    if model_name not in model_loader.models:
        return "Model không hợp lệ", [], 0

    full_model = model_loader.models[model_name]
    extractor = model_loader.extractors[model_name]

    # Phân loại ảnh đầu vào
    image_array = preprocess_image(input_image_path)
    predicted_class = classify_image(full_model, image_array)
    print(f"Phân lớp ảnh đầu vào: {predicted_class}")

    # Trích xuất đặc trưng của ảnh đầu vào
    input_feature = extract_feature(input_image_path, extractor)
    print(f"Kích thước đặc trưng ảnh đầu vào: {input_feature.shape}")

    # Kết nối cơ sở dữ liệu
    conn = connect_db()
    cursor = conn.cursor()

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
                # Đọc file ảnh và chuyển thành base64
                img_path = os.path.join(IMAGE_ROOT_DIR, predicted_class, image_name)
                if os.path.exists(img_path):
                    with open(img_path, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        img_base64 = f"data:image/png;base64,{img_data}"
                else:
                    print(f"Không tìm thấy file ảnh: {img_path}")
                    img_base64 = ""

                similar_images.append({
                    'image_id': image_id,
                    'image_name': image_name,
                    'similarity': round(float(similarity), 4),
                    'image_data': img_base64,
                    'doi': doi,
                    'title': title,
                    'caption': caption,
                    'authors': authors
                })
        except Exception as e:
            print(f"Lỗi với ảnh {image_name}: {e}")
            continue

    similar_images.sort(key=lambda x: x['similarity'], reverse=True)
    total_similar_images = len(similar_images)  # Tổng số ảnh tương đồng

    cursor.close()
    conn.close()

    return predicted_class, similar_images, total_similar_images

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

# Hàm chính để kiểm tra từ terminal
def main():
    input_image_path = r"C:/Users/anh32/Downloads/CT551/CT551/Animal Samples/01-CN-HOANG THI NGHIEP(1-5)_page_3_img_1.png"
    model_name = "MobileNetV2"
    threshold = 0.95

    predicted_class, similar_images, total_similar_images = compute_similarity(input_image_path, model_name, threshold)

    print(f"Phân lớp ảnh: {predicted_class}")
    print(f"Tổng số ảnh tương đồng: {total_similar_images}")
    print("Các ảnh tương đồng:")
    if not similar_images:
        print("Không tìm thấy ảnh nào tương đồng với ngưỡng hiện tại.")
    else:
        for img in similar_images:
            print(f"- {img['image_name']} (ID: {img['image_id']}): {img['similarity']}")
            if img['image_data']:
                print(f"  (Ảnh đã được tìm thấy và chuyển thành base64)")
            else:
                print(f"  (Không tìm thấy ảnh trong thư mục)")

if __name__ == "__main__":
    # Chạy Flask server
    app.run(debug=True, host='0.0.0.0', port=5001)