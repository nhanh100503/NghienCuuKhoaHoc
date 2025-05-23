from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
import os
from dotenv import load_dotenv
load_dotenv()
import os
from flask import send_from_directory

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

CORS(app, resources={r"/classify": {"origins": "http://localhost:5173"}})

MODEL_FILES = {
    'model1': 'E:/NCKH/MODEL/ResNet101/resnet101.keras',
    'model2': 'E:/NCKH/MODEL/ResNet101/tc-resnet101.keras',
    'model3': 'E:/NCKH/MODEL/MobileNetV2/mobilenetv2.keras',
    'model4': 'E:/NCKH/MODEL/MobileNetV2/tc-mobilenetv2.keras',
    'model5': 'E:/NCKH/MODEL/EfficientNetB0/efficientnetb0.keras',  
    'model6': 'E:/NCKH/MODEL/EfficientNetB0/tc-efficientnetb0.keras'    
}

CLASS_INDICES_DIR = 'E:/NCKH/DATASET/raw_dataset/train'

IMAGE_SIZE = (224, 224)



RAW_DATA = os.getenv('RAW_DATASET')

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(RAW_DATA, filename)


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

def load_resources():
    class_indices = load_class_indices()
    models = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            try:
                models[name] = load_model(path)
                print(f"Loaded model: {name}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        else:
            print(f"Model path not found: {path}")

    return models, class_indices

models_dict, class_indices = load_resources()

def predict_image_class(image_stream, model, class_indices, target_size=(224, 224)):
    try:
        img = Image.open(image_stream).convert('RGB')
        img = img.resize(target_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Lỗi xử lý ảnh: {str(e)}")

    prediction = model.predict(img_array)
    probabilities = prediction[0]

    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    top_5_results = []
    for idx in top_5_indices:
        class_name = list(class_indices.keys())[list(class_indices.values()).index(idx)]
        confidence = float(probabilities[idx])
        top_5_results.append({'class': class_name, 'confidence': confidence})

    return top_5_results

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'Thiếu ảnh hoặc tên model'}), 400

    image_file = request.files['image']
    model_name = request.form['model']

    if image_file.filename == '':
        return jsonify({'error': 'Không có ảnh được chọn'}), 400

    if model_name not in MODEL_FILES or model_name not in models_dict:
        return jsonify({'error': f'Model không hợp lệ: {model_name}'}), 400

    try:
        results = predict_image_class(image_file.stream, models_dict[model_name], class_indices, target_size=IMAGE_SIZE)
        return jsonify({'model': model_name, 'predictions': results})
    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý ảnh: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
