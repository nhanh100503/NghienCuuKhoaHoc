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
from dotenv import load_dotenv
load_dotenv()
import time
import numpy as np
import json
import matplotlib.pyplot as plt
import os
RAW_DATASET = os.getenv('RAW_DATASET')

# --------- Cấu hình DB -------------
db_config = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'database': 'nckh',
}

# --------- Hàm kết nối và lấy dữ liệu feature và nhãn ---------
def load_features_and_labels(model_name=''):
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





class_names = [folder for folder in os.listdir(RAW_DATASET) if os.path.isdir(os.path.join(RAW_DATASET, folder))]

from sklearn.metrics.pairwise import euclidean_distances

def compute_similarity(query_vec, db_features):
    """
    Hàm tính similarity giữa query_vec và toàn bộ db_features.
    metric: 'cosine' hoặc 'euclidean'
    Trả về mảng similarity hoặc khoảng cách.
    """
    # Chuẩn hóa vector
    q = query_vec / np.linalg.norm(query_vec)
    db_norm = db_features / np.linalg.norm(db_features, axis=1, keepdims=True)
    sims = np.dot(db_norm, q)
    return sims
   



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







def compute_map_with_threshold(test_features, test_labels, db_features, db_labels, thresholds, faiss_index=None, model_name=''):
    import time
    import json
    import numpy as np

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
        aps, precisions, recalls, f1s = [], [], [], []
        total_query_time = 0

        for query_vec, query_label in zip(test_features, test_labels):
            start_time = time.time()

            if faiss_index is not None:
                # FAISS yêu cầu query phải được reshape và normalize
                query_vec = query_vec.reshape(1, -1)
                faiss.normalize_L2(query_vec)

                sims, indices = faiss_index.search(query_vec, db_features.shape[0])
                filtered_pairs = [(i, s) for i, s in zip(indices[0], sims[0]) if s >= t]

            else:
                sims = compute_similarity(query_vec, db_features)

                filtered_pairs = [(i, s) for i, s in enumerate(sims) if s >= t]

            if not filtered_pairs:
                aps.append(0)
                precisions.append(1.0)
                recalls.append(0.0)
                f1s.append(0.0)
                total_query_time += time.time() - start_time
                continue

            filtered_indices = [i for i, _ in filtered_pairs]
            filtered_sims = [s for _, s in filtered_pairs]
            filtered_labels = [db_labels[i] for i in filtered_indices]

            # Sắp xếp theo similarity giảm dần
            sorted_idx = np.argsort([-s for s in filtered_sims])
            sorted_labels = [filtered_labels[i] for i in sorted_idx]
            y_true = [1 if lbl == query_label else 0 for lbl in sorted_labels]

            ap = average_precision(y_true)
            aps.append(ap)

            tp = sum(y_true)
            total_relevant = sum(lbl == query_label for lbl in db_labels)
            precision = tp / len(y_true)
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

    print(f"\n✅ Ngưỡng tốt nhất theo mAP: {best_threshold:.2f} với mAP = {best_map:.4f}")

    # Lưu kết quả ra file
    with open(f"{model_name}_map_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    return best_threshold, best_map



def euclidean_to_similarity(dists, gamma=1.0):
    return np.exp(-gamma * (np.array(dists) ** 2))


def load_index_and_ids(model_name='', class_names=class_names):
    all_image_ids = []
    all_labels = []
    all_features = []

    if class_names is None:
        class_names = [f.split("_image_ids_")[1].replace(".pkl", "") 
                       for f in os.listdir(INDEX_DIR) if f.startswith(f"{model_name}_image_ids_")]

    index_combined = None

    for class_name in class_names:
        index_path = os.path.join(INDEX_DIR, f"{model_name}_class_{class_name}.index")
        ids_path = os.path.join(INDEX_DIR, f"{model_name}_image_ids_{class_name}.pkl")

        index = faiss.read_index(index_path)
        image_ids = joblib.load(ids_path)

        vectors = index.reconstruct_n(0, index.ntotal)
        vectors = normalize(vectors, norm='l2')  # ✅ Chuẩn hóa trước khi add

        all_image_ids.extend(image_ids)
        all_labels.extend([class_name] * len(image_ids))
        all_features.append(vectors)

        if index_combined is None:
            index_combined = faiss.IndexFlatIP(vectors.shape[1])  # ✅ Inner Product
        index_combined.add(vectors)  # ✅ Đã normalize xong thì add

        print(f"✅ Loaded {len(image_ids)} vectors for class: {class_name}")

    features = np.vstack(all_features)

    return all_image_ids, features, all_labels, index_combined





def plot_results_from_files(model_names):
    results = {}

    for model_name in model_names:
        file_path = fr"E:\NghienCuuKhoaHoc\INTERFACE\backend\{model_name}_map_results.json"
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
INDEX_DIR = r"E:\NghienCuuKhoaHoc\INDEX\resnet101"
# INDEX_DIR = r"E:\NghienCuuKhoaHoc\INDEX\efficientnetb0"
# INDEX_DIR = r"E:\NghienCuuKhoaHoc\INDEX\mobilenetv2"

# ------------Faiss---------------
if __name__ == "__main__":
    model_name = 'ResNet101'
    metric = 'cosine'  # hoặc 'euclidean'

    # Faiss
    image_ids_db, features_db, labels_db, index_combined = load_index_and_ids(model_name=model_name)


    model_loader = ModelLoader()
    extractor = model_loader.extractors[model_name]

    test_features = []
    test_labels = []
    for img_path, true_label in test_images:

        query_feat = extract_feature(img_path, extractor, model_name=model_name, size=(224, 224)).astype('float32')

        
        query_feat = query_feat.reshape(1, -1)  # reshape thành 2D cho faiss.normalize_L2
        faiss.normalize_L2(query_feat)  # Chuẩn hóa vector truy vấn

        test_features.append(query_feat[0])  # Append vector 1D, không phải 2D
        test_labels.append(true_label)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    best_threshold, best_map = compute_map_with_threshold(
        test_features, test_labels, features_db, labels_db, thresholds, faiss_index=index_combined, model_name='ResNet101 Faiss'
    )
    print(f"Ngưỡng tốt nhất theo mAP: {best_threshold:.2f} với mAP = {best_map:.4f}")

# if __name__ == "__main__":
#     model_name = 'ResNet101'
#     metric = 'cosine'  # hoặc 'euclidean'


#     # Không Faiss
#     image_ids_db, features_db, labels_db = load_features_and_labels(model_name='ResNet101')

#     model_loader = ModelLoader()
#     extractor = model_loader.extractors[model_name]

#     test_features = []
#     test_labels = []
#     for img_path, true_label in test_images:

#         query_feat = extract_feature(img_path, extractor=extractor, model_name=model_name, size=(224, 224))

#         test_features.append(query_feat)
#         test_labels.append(true_label)

#     thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
#     best_threshold, best_map = compute_map_with_threshold(
#         test_features, test_labels, features_db, labels_db, thresholds, faiss_index=None, model_name='ResNet101'
#     )
#     print(f"Ngưỡng tốt nhất theo mAP: {best_threshold:.2f} với mAP = {best_map:.4f}")




# plot_results_from_files(['ResNet101', 'MobileNetV2', 'EfficientNetB0'])