import numpy as np
import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score
from app_similarity import *
import os
import faiss
import joblib
import numpy as np
import numpy as np
import time
import numpy as np
import json
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


import json
import matplotlib.pyplot as plt
import os

# --------- Cấu hình DB -------------
db_config = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'database': 'nckh',
}

# --------- Hàm kết nối và lấy dữ liệu feature và nhãn ---------
def load_features_and_labels(model_name='InceptionV4_TC'):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    query = f"""
    SELECT f.image_id, f.feature_vector, r.class_name
    FROM feature f
    JOIN research r ON f.image_id = r.image_id
    WHERE f.model_name = %s
    """
    cursor.execute(query, (model_name,))

    image_ids = []
    features = []
    labels = []
    for image_id, feature_blob, class_name in cursor:


        feat_str = feature_blob.decode('utf-8')  # chuyển bytes -> string
        feature_vec = np.array(list(map(float, feat_str.split(','))), dtype=np.float32)  # string -> float array

        image_ids.append(image_id)
        features.append(feature_vec)
        labels.append(class_name)

    cursor.close()
    conn.close()

    features = np.vstack(features)
    return image_ids, features, labels



INDEX_DIR = r"E:\NghienCuuKhoaHoc\INDEX\convnext_v2"


class_names = [folder for folder in os.listdir(RAW_DATASET) if os.path.isdir(os.path.join(RAW_DATASET, folder))]

from sklearn.metrics.pairwise import euclidean_distances

def compute_similarity(query_vec, db_features, metric=''):
    """
    Hàm tính similarity giữa query_vec và toàn bộ db_features.
    metric: 'cosine' hoặc 'euclidean'
    Trả về mảng similarity hoặc khoảng cách.
    """
    if metric == 'cosine':
        # Chuẩn hóa vector
        q = query_vec / np.linalg.norm(query_vec)
        db_norm = db_features / np.linalg.norm(db_features, axis=1, keepdims=True)
        sims = np.dot(db_norm, q)
        return sims
    elif metric == 'euclidean':
        # Khoảng cách Euclidean
        diff = db_features - query_vec
        dists = np.linalg.norm(diff, axis=1)
        # dists = euclidean_distances(features_db, query_feat.reshape(1, -1)).flatten()
        return dists
    else:
        raise ValueError("Metric phải là 'cosine' hoặc 'euclidean'")



def load_test_images_from_folder(test_root):
    test_images = []
    # Duyệt từng thư mục con (tên thư mục = label)
    for class_name in os.listdir(test_root):
        class_dir = os.path.join(test_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        # Duyệt các ảnh trong thư mục này
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(class_dir, filename)
                test_images.append((img_path, class_name))
    return test_images
# Ví dụ gọi
test_root = r'E:\NghienCuuKhoaHoc\DATASET\split_dataset\test'
test_images = load_test_images_from_folder(test_root)
print(f"Tổng số ảnh test: {len(test_images)}") 


convnext_v2_path = r'E:\NghienCuuKhoaHoc\MODEL\convnext_v2\convnext_v2_best_params_aug_final.pth'
convnext_v2 = load_convnextv2_model(model_path=convnext_v2_path, num_classes=11, model_type= 'convnext_v2_aug')
vgg16 = load_model(r'E:\NghienCuuKhoaHoc\MODEL\vgg16\vgg16_aug_best_params_final.keras', compile=False)



import numpy as np

def average_precision(y_true):
    """
    Tính Average Precision cho danh sách nhãn y_true (1 - relevant, 0 - non-relevant),
    theo thứ tự đã sắp xếp (tương tự sklearn.metrics.average_precision_score).
    """
    y_true = np.array(y_true)
    relevant = y_true == 1
    if relevant.sum() == 0:
        return 0.0

    precisions = []
    num_relevant = 0
    for i, rel in enumerate(relevant, start=1):
        if rel:
            num_relevant += 1
            precisions.append(num_relevant / i)

    return np.mean(precisions)


import time
import numpy as np




def minmax_euclidean_to_similarity(dists):
    dists = np.array(dists)
    d_min = dists.min()
    d_max = dists.max()
    normalized = (dists - d_min) / (d_max - d_min + 1e-8)  # tránh chia 0
    similarity = 1 - normalized
    return similarity

def compute_map_with_threshold(test_features, test_labels, db_features, db_labels, thresholds, metric='', model_name=''):
    best_map = 0
    best_threshold = None

    results_dict = {
        'thresholds': [],
        'maps': [],
        'precisions': [],
        'recalls': [],
        'f1s': [],
        'times': [],
        'model': model_name
    }

    for t in thresholds:
        aps = []
        precisions = []
        recalls = []
        f1s = []
        total_query_time = 0

        for query_vec, query_label in zip(test_features, test_labels):
            start_time = time.time()

            sims = compute_similarity(query_vec, db_features, metric)

            if metric == 'euclidean':
                # filtered_indices = [i for i, s in enumerate(sims) if s <= t]
                        # Chuyển distance thành similarity

                gamma = 1.0  # bạn có thể chỉnh giá trị này để tối ưu kết quả

                sims = np.exp(-gamma * (np.array(sims) ** 2))

                filtered_indices = [i for i, s in enumerate(sims) if s >= t]

            else:
                filtered_indices = [i for i, s in enumerate(sims) if s >= t]

            if not filtered_indices:
                aps.append(0)
                precisions.append(1.0)  # Không chọn ảnh nào, precision = 1
                recalls.append(0.0)
                f1s.append(0.0)
                total_query_time += time.time() - start_time
                continue

            filtered_labels = [db_labels[i] for i in filtered_indices]
            filtered_sims = [sims[i] for i in filtered_indices]

            if metric == 'euclidean':
                # sorted_idx = np.argsort(filtered_sims)  # tăng dần
                sorted_idx = np.argsort([-s for s in filtered_sims])  # luôn giảm dần

            else:
                sorted_idx = np.argsort([-s for s in filtered_sims])  # giảm dần

            sorted_labels = [filtered_labels[i] for i in sorted_idx]
            y_true = [1 if lbl == query_label else 0 for lbl in sorted_labels]

            ap = average_precision(y_true)
            aps.append(ap)

            tp = sum(y_true)
            total_relevant = sum(lbl == query_label for lbl in db_labels)
            precision = tp / len(y_true) if len(y_true) > 0 else 1.0
            recall = tp / total_relevant if total_relevant > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            total_query_time += time.time() - start_time

        mean_ap = np.mean(aps)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1 = np.mean(f1s)
        avg_query_time = total_query_time / len(test_features)

        print(f"Ngưỡng {t:.2f} => mAP: {mean_ap:.4f}, Precision: {mean_precision:.4f}, "
              f"Recall: {mean_recall:.4f}, F1: {mean_f1:.4f}, Avg Time: {avg_query_time:.4f}s")

        results_dict['thresholds'].append(t)
        results_dict['maps'].append(mean_ap)
        results_dict['precisions'].append(mean_precision)
        results_dict['recalls'].append(mean_recall)
        results_dict['f1s'].append(mean_f1)
        results_dict['times'].append(avg_query_time)

        if mean_ap > best_map:
            best_map = mean_ap
            best_threshold = t

    print(f"\nNgưỡng tốt nhất theo mAP: {best_threshold:.2f} với mAP = {best_map:.4f}")

    # Lưu kết quả ra file JSON
    with open(f"{model_name}_map_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    return best_threshold, best_map


def euclidean_to_similarity(dists, gamma=1.0):
    return np.exp(-gamma * (np.array(dists) ** 2))

def load_index_and_ids(model_name='convnext_v2_aug', class_names=class_names):
    all_image_ids = []
    all_labels = []
    all_features = []

    if class_names is None:
        # Nếu không truyền vào, bạn có thể tự liệt kê tất cả class_names đã lưu
        class_names = [f.split("_image_ids_")[1].replace(".pkl", "") 
                       for f in os.listdir(INDEX_DIR) if f.startswith(f"{model_name}_image_ids_")]

    for class_name in class_names:
        index_path = os.path.join(INDEX_DIR, f"{model_name}_class_{class_name}.index")
        ids_path = os.path.join(INDEX_DIR, f"{model_name}_image_ids_{class_name}.pkl")

        # Load FAISS index
        index = faiss.read_index(index_path)

        # Load image_ids
        image_ids = joblib.load(ids_path)

        # Lấy vector từ index
        vectors = index.reconstruct_n(0, index.ntotal)  # hoặc dùng index.xb nếu là IndexFlat

        # Gộp vào danh sách tổng
        all_image_ids.extend(image_ids)
        all_labels.extend([class_name] * len(image_ids))
        all_features.append(vectors)

        print(f"✅ Loaded {len(image_ids)} vectors for class: {class_name}")

    # Nối tất cả vectors lại thành 1 mảng
    features = np.vstack(all_features)

    return all_image_ids, features, all_labels


def plot_results_from_files(model_names):
    results = {}

    for model_name in model_names:
        file_path = fr"E:\NghienCuuKhoaHoc\INTERFACE\backend\euclidean\{model_name}_map_results.json"
        with open(file_path, "r") as f:
            results[model_name] = json.load(f)

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    plt.figure(figsize=(14, 5))

    # ----------- Biểu đồ mAP theo ngưỡng ------------
    plt.subplot(1, 2, 1)
    for i, (model_name, data) in enumerate(results.items()):
        thresholds = data['thresholds']
        maps = data['maps']
        marker = markers[i % len(markers)]
        plt.plot(thresholds, maps, marker=marker, label=model_name)
    plt.title('mAP theo ngưỡng')
    plt.xlabel('Ngưỡng')
    plt.ylabel('mAP')
    # plt.grid(True)
    plt.legend()
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    base_thresholds = results[model_names[0]]['thresholds']
    plt.xticks(base_thresholds)

    # ----------- Biểu đồ thời gian theo ngưỡng ------------
    plt.subplot(1, 2, 2)
    for i, (model_name, data) in enumerate(results.items()):
        thresholds = data['thresholds']
        times_ms = [t * 1000 for t in data['times']]
        marker = markers[i % len(markers)]
        plt.plot(thresholds, times_ms, marker=marker, label=model_name)
    plt.title('Thời gian truy vấn trung bình theo ngưỡng')
    plt.xlabel('Ngưỡng')
    plt.ylabel('Thời gian (ms)')
    # plt.grid(True)
    plt.legend()
    plt.xticks(base_thresholds)

    plt.tight_layout()
    plt.show()



from sklearn.preprocessing import normalize

if __name__ == "__main__":
    model_name = 'ResNet101'
    metric = 'cosine'  # hoặc 'euclidean'

    # image_ids_db, features_db, labels_db = load_index_and_ids(model_name=model_name)

    image_ids_db, features_db, labels_db = load_features_and_labels(model_name='ResNet101')
    model_loader = ModelLoader()
    extractor = model_loader.extractors[model_name]
    # features_db = normalize(features_db, norm='l2')

    test_features = []
    test_labels = []
    for img_path, true_label in test_images:

        # img = Image.open(img_path).convert("RGB")
        # load_pca(model_type='convnext_v2')
        # query_feat = extract_spoc_features(img, model=convnext_v2, model_type=model_name)


        query_feat = extract_feature(img_path, extractor=extractor, model_name=model_name, size=(224, 224))
        # query_feat = normalize(query_feat.reshape(1, -1), norm='l2')[0]

        # query_vector = apply_pca(query_feat)
        test_features.append(query_feat)
        test_labels.append(true_label)
        

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    best_threshold, best_map = compute_map_with_threshold(
        test_features, test_labels, features_db, labels_db, thresholds, metric=metric, model_name='ResNet101'
    )
    print(f"Ngưỡng tốt nhất theo mAP: {best_threshold:.2f} với mAP = {best_map:.4f}")

# plot_results_from_files(['VGG16', 'ConvNeXt V2', 'ResNet101', 'Inception V4'])
