# # # import torch
# # # from PIL import Image
# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # from torchvision import transforms
# # # from load_model import *
# # # import torch.nn.functional as F

# # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # # # Tiền xử lý ảnh
# # # preprocess = transforms.Compose([
# # #     transforms.Resize((224, 224)),
# # #     transforms.ToTensor(),
# # #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
# # #                          std=[0.229, 0.224, 0.225]),
# # # ])

# # # def extract_feature_maps_per_stage(model, img_tensor):
# # #     features = []
# # #     with torch.no_grad():
# # #         x = model.stem(img_tensor)
# # #         features.append(x)  # Đặc trưng sau stem (coi như stage -1)

# # #         for stage in model.stages:
# # #             x = stage(x)
# # #             features.append(x)  # Đặc trưng sau mỗi stage
# # #     return features  # List of [1, C, H, W]

# # # def extract_all_stage_feature_maps(model, img_tensor):
# # #     feature_maps = {}
# # #     with torch.no_grad():
# # #         x = model.stem(img_tensor)
# # #         feature_maps['stem'] = x
# # #         for i, stage in enumerate(model.stages):
# # #             x = stage(x)
# # #             feature_maps[f'stage{i+1}'] = x
# # #     return feature_maps


# # # def plot_best_feature_per_stage(feature_list):
# # #     num_stages = len(feature_list)
# # #     plt.figure(figsize=(num_stages * 3, 3))

# # #     for i, fmap in enumerate(feature_list):
# # #         fmap = fmap.squeeze(0)  # [C, H, W]
# # #         avg_activations = fmap.mean(dim=(1, 2))  # [C]
# # #         best_idx = torch.argmax(avg_activations).item()

# # #         fmap_i = fmap[best_idx].cpu().numpy()
# # #         fmap_norm = (fmap_i - fmap_i.min()) / (fmap_i.max() - fmap_i.min() + 1e-10)

# # #         ax = plt.subplot(1, num_stages, i + 1)
# # #         plt.imshow(fmap_norm, cmap='viridis')
# # #         plt.axis('off')
# # #         ax.set_title(f"Stage {i-1 if i > 0 else 'Stem'}\nKênh {best_idx}", fontsize=10)

# # #     plt.tight_layout()
# # #     plt.show()



# # # # Hiển thị tất cả đặc trưng từ ảnh
# # # def visualize_best_feature_per_stage(img_path, model):
# # #     img = Image.open(img_path).convert('RGB')
# # #     img_tensor = preprocess(img).unsqueeze(0).to(device)

# # #     features = extract_feature_maps_per_stage(model, img_tensor)

# # #     # Resize tất cả về 224x224 để dễ xem
# # #     resized_features = [F.interpolate(f, size=(224, 224), mode='bilinear', align_corners=False)
# # #                         for f in features]

# # #     plot_best_feature_per_stage(resized_features)


# # # # Load model ConvNeXt
# # # model_convnext_v2 = load_convnextv2_model(
# # #     r'E:\NghienCuuKhoaHoc\MODEL\convnext_v2\convnext_v2_best_params_aug_final.pth',
# # #     11,
# # #     "convnext_v2_aug"
# # # )
# # # model_convnext_v2.to(device)
# # # model_convnext_v2.eval()

# # # # Đường dẫn ảnh
# # # img_path = 'E:/LuanVan/data/raw/Maps/04-BE-VO NAM SON(24-38)004_page_26_img_1.png'

# # # # Hiển thị toàn bộ feature maps (của layer đầu tiên)
# # # visualize_best_feature_per_stage(img_path, model_convnext_v2)

# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# from torchvision import transforms
# import torch.nn.functional as F
# from load_model import *
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Tiền xử lý ảnh
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# # Trích xuất feature maps từ tất cả các stage (stem + 4 stages)
# def extract_feature_maps_all_stages(model, img_tensor):
#     with torch.no_grad():
#         features = []
#         x = model.stem(img_tensor)
#         features.append(x)
#         for stage in model.stages:
#             x = stage(x)
#             features.append(x)
#     return features  # List các tensor [1, C, H, W]

# # Chuẩn hóa riêng từng feature map để scale về [0, 1]
# def normalize_feature_map(fmap):
#     fmap = fmap - fmap.min()
#     fmap = fmap / (fmap.max() + 1e-10)
#     return fmap

# # Lấy top k channels theo tổng giá trị tuyệt đối (activation mạnh nhất)
# def get_topk_channels(fmap, k=10):
#     fmap_abs = fmap.abs()
#     channel_sums = fmap_abs.sum(dim=[2,3])  # [1, C]
#     topk_vals, topk_idx = torch.topk(channel_sums, k, dim=1)
#     return topk_idx[0].cpu().numpy()  # Trả về indices

# # Hiển thị các channel đã chọn dưới dạng lưới
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from PIL import Image
# import torch.nn.functional as F

# def visualize_top2_channels_each_stage_including_stem(img_path, model):
#     # Đọc ảnh gốc
#     img = Image.open(img_path).convert('RGB')
#     img_tensor = preprocess(img).unsqueeze(0).to(device)

#     # Lấy feature maps các stage (dạng list tensor: [1, C, H, W])
#     features = extract_feature_maps_all_stages(model, img_tensor)

#     # Resize tất cả feature maps về 224x224 cho đồng bộ hiển thị
#     # features_resized = [F.interpolate(fmap, size=(224, 224), mode='bilinear', align_corners=False) for fmap in features]
#     features_resized = features
#     top_feature_maps = []
#     titles = []

#     for stage_idx, fmap in enumerate(features_resized):  # Duyệt từng stage
#         fmap_abs = fmap.abs()  # [1, C, H, W]
#         channel_sum = fmap_abs.sum(dim=[2, 3]).squeeze(0)  # Tổng kích hoạt mỗi channel [C]

#         # Lấy 2 channel có kích hoạt cao nhất
#         top2_idx = torch.topk(channel_sum, k=2).indices.tolist()

#         stage_name = "Conv" if stage_idx == 0 else f"Stage {stage_idx}"

#         for ch_idx in top2_idx:
#             fmap_norm = normalize_feature_map(fmap[0, ch_idx].cpu())
#             top_feature_maps.append(fmap_norm)
#             titles.append(f"{stage_name}")  # Thêm chỉ số channel vào tiêu đề
#             print(f"[{stage_name}] Top channel: {ch_idx}, Score: {channel_sum[ch_idx]:.4f}")

#     # Thiết lập grid vẽ ảnh gốc bên trái, feature maps bên phải
#     num_maps = len(top_feature_maps)
#     num_cols = 5
#     num_rows = (num_maps + num_cols - 1) // num_cols
#     rows_for_img_col = max(num_rows, 3)  # Ít nhất 3 hàng cho cột ảnh gốc

#     plt.figure(figsize=(3 * (num_cols + 1), max(3, rows_for_img_col * 3)))
#     gs = gridspec.GridSpec(nrows=rows_for_img_col, ncols=num_cols + 1, width_ratios=[1.5] + [1]*num_cols)  # width_ratios chỉnh 1.5 như TF code

#     # Vẽ ảnh gốc chiếm 2 hàng ở giữa cột 0
#     middle_row = rows_for_img_col // 2
#     ax_img = plt.subplot(gs[middle_row-1:middle_row+1, 0])
#     ax_img.imshow(img)
#     ax_img.set_title("Ảnh gốc", fontsize=12)
#     ax_img.axis('off')
#     # Bỏ dòng set_aspect('auto') để giống TF code
#     # ax_img.set_aspect('auto')

#     # Vẽ các feature maps ở cột còn lại
#     for i, fmap in enumerate(top_feature_maps):
#         row = i // num_cols
#         col = i % num_cols + 1  # +1 vì cột 0 dành ảnh gốc
#         ax = plt.subplot(gs[row, col])
#         ax.imshow(fmap, cmap='viridis')
#         ax.set_title(titles[i], fontsize=10)
#         ax.axis('off')
#         ax.set_aspect('equal')

#     plt.tight_layout()
#     plt.show()

# # Gọi hàm mới


# def plot_feature_maps_grid_concat(fmaps_list, titles_list, max_cols=5, cmap='viridis'):
#     num_maps = len(fmaps_list)
#     num_cols = min(max_cols, num_maps)
#     num_rows = (num_maps + num_cols - 1) // num_cols

#     plt.figure(figsize=(num_cols * 3, num_rows * 3))
#     for i, fmap in enumerate(fmaps_list):
#         ax = plt.subplot(num_rows, num_cols, i + 1)
#         plt.imshow(fmap.cpu(), cmap=cmap)
#         plt.title(titles_list[i], fontsize=9)
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()


# model_convnext_v2 = load_convnextv2_model(
#     r'E:\NghienCuuKhoaHoc\MODEL\convnext_v2\convnext_v2_best_params_aug_final.pth',
#     num_classes=11,
#     model_type="convnext_v2_aug"
# )
# model_convnext_v2.to(device)
# model_convnext_v2.eval()

# img_path = 'E:/LuanVan/data/raw/Maps/04-BE-VO NAM SON(24-38)004_page_26_img_1.png'

# visualize_top2_channels_each_stage_including_stem(img_path, model_convnext_v2)
# # import tensorflow as tf
# # from tensorflow.keras.models import load_model, Model
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # Load model đã huấn luyện
# # model = load_model(r'E:\NghienCuuKhoaHoc\MODEL\resnet101\tc-resnet101.keras', compile=False)

# # layer_names = [
# #     'conv1_relu',          # Stem (không phải block chính)
# #     'conv2_block3_out',    # Layer 1 (Block 1)
# #     'conv3_block4_out',    # Layer 2 (Block 2)
# #     'conv4_block23_out',   # Layer 3 (Block 3)
# #     'conv5_block3_out'     # Layer 4 (Block 4)
# # ]

# # # Đặt tên hiển thị tương ứng cho từng layer
# # display_layer_names = [
# #     'Conv',   # Stem
# #     'Layer 1 (conv2_block3_out)',  
# #     'Layer 2 (conv3_block4_out)',  
# #     'Layer 3 (conv4_block23_out)', 
# #     'Layer 4 (conv5_block3_out)'
# # ]
# # # Model trích xuất feature maps
# # feature_extractor = Model(
# #     inputs=model.input,
# #     outputs=[model.get_layer(name).output for name in layer_names]
# # )

# # # Tiền xử lý ảnh
# # def preprocess_tf_image(img_path):
# #     img = tf.io.read_file(img_path)
# #     img = tf.image.decode_png(img, channels=3)
# #     img = tf.image.resize(img, (224, 224))
# #     img = img / 255.0
# #     img = tf.expand_dims(img, axis=0)
# #     return img

# # # Chuẩn hóa về [0,1]
# # def normalize_feature_map(fmap):
# #     return (fmap - np.min(fmap)) / (np.max(fmap) - np.min(fmap) + 1e-10)

# # # ✅ Hàm lấy top N channels theo từng layer
# # def get_topk_per_layer(feature_maps, k=2):
# #     """
# #     Trả về danh sách (layer_idx, channel_idx) top k channel có mean activation cao nhất mỗi layer
# #     """
# #     topk = []
# #     for layer_idx, fmap in enumerate(feature_maps):
# #         fmap_abs = np.abs(fmap)
# #         means = fmap_abs.mean(axis=(0, 1))  # mean theo channel
# #         top_indices = np.argsort(means)[-k:][::-1]
# #         for ch_idx in top_indices:
# #             topk.append((layer_idx, ch_idx))
# #     return topk

# # import matplotlib.pyplot as plt
# # import matplotlib.gridspec as gridspec

# # def visualize_topk_per_layer(img_path, feature_extractor, top_per_layer=2):
# #     # Tiền xử lý ảnh đầu vào (theo TF)
# #     img_tensor = preprocess_tf_image(img_path)  # giả sử trả về batch 1 tensor

# #     # Lấy feature maps (list tensor [1, H, W, C] hoặc [1, C, H, W], tùy model)
# #     features = feature_extractor(img_tensor)

# #     # Chuyển sang numpy và reshape về [H, W, C] hoặc [C, H, W] tuỳ model
# #     # Ở đây mình giả sử features là list tensor [1, H, W, C]
# #     feature_maps_np = [fmap.numpy()[0] for fmap in features]

# #     # Lấy top k channels per layer, trả về list [(layer_idx, channel_idx), ...]
# #     topk = get_topk_per_layer(feature_maps_np, k=top_per_layer)

# #     topk_feature_maps = []
# #     titles = []
# #     for layer_idx, ch_idx in topk:
# #         # Lấy channel tương ứng, giả sử channel cuối cùng
# #         fmap = feature_maps_np[layer_idx][:, :, ch_idx]  # [H, W]
# #         topk_feature_maps.append(fmap)
# #         titles.append(f"{display_layer_names[layer_idx]}")

# #     # Đọc và resize ảnh gốc để hiển thị
# #     img_orig = tf.io.read_file(img_path)
# #     img_orig = tf.image.decode_png(img_orig, channels=3)
# #     img_orig = tf.image.resize(img_orig, (224, 224))
# #     img_orig = img_orig.numpy().astype(np.uint8)

# #     num_maps = len(topk_feature_maps)
# #     num_cols = 5
# #     num_rows = (num_maps + num_cols - 1) // num_cols
# #     rows_for_img_col = max(num_rows, 3)  # Ít nhất 3 hàng cho cột ảnh gốc

# #     plt.figure(figsize=(3 * (num_cols + 1), max(3, rows_for_img_col * 3)))

# #     gs = gridspec.GridSpec(nrows=rows_for_img_col, ncols=num_cols + 1, width_ratios=[1.5] + [1]*num_cols)

# #     # Vẽ ảnh gốc chiếm 2 hàng ở giữa cột 0
# #     middle_row = rows_for_img_col // 2
# #     ax_img = plt.subplot(gs[middle_row-1:middle_row+1, 0])
# #     ax_img.imshow(img_orig)
# #     ax_img.set_title("Ảnh gốc", fontsize=12)
# #     ax_img.axis('off')
# #     # Bỏ set_aspect('auto') để tránh bóp méo

# #     # Vẽ các feature maps ở cột còn lại
# #     for i, fmap in enumerate(topk_feature_maps):
# #         row = i // num_cols
# #         col = i % num_cols + 1
# #         ax = plt.subplot(gs[row, col])
# #         fmap_norm = normalize_feature_map(fmap)
# #         ax.imshow(fmap_norm, cmap='viridis')
# #         ax.set_title(titles[i], fontsize=10)
# #         ax.axis('off')
# #         ax.set_aspect('equal')

# #     plt.tight_layout()
# #     plt.show()


# # # Hàm hiển thị lưới ảnh
# # def plot_feature_maps_grid_concat(fmaps_list, titles_list, max_cols=5, cmap='viridis'):
# #     num_maps = len(fmaps_list)
# #     num_cols = min(max_cols, num_maps)
# #     num_rows = (num_maps + num_cols - 1) // num_cols

# #     plt.figure(figsize=(num_cols * 3, num_rows * 3))
# #     for i, fmap in enumerate(fmaps_list):
# #         ax = plt.subplot(num_rows, num_cols, i + 1)
# #         fmap_norm = normalize_feature_map(fmap)
# #         plt.imshow(fmap_norm, cmap=cmap)
# #         plt.title(titles_list[i], fontsize=10)
# #         plt.axis('off')
# #     plt.tight_layout()
# #     plt.show()

# # # 📷 Đường dẫn ảnh
# # img_path = 'E:/LuanVan/data/raw/Maps/04-BE-VO NAM SON(24-38)004_page_26_img_1.png'

# # # 🔍 Hiển thị top 2 mỗi layer
# # visualize_topk_per_layer(img_path, feature_extractor, top_per_layer=2)
# # --- 1. Import các thư viện cần thiết ---
# # import torch
# # import torch.nn.functional as F
# # import torchvision.transforms as transforms
# # from PIL import Image
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import timm
# # from load_model import *

# # # --- 1. Load mô hình InceptionV4 từ file .pth ---
# # model = load_inceptionv4_feature_extractor(r'E:\NghienCuuKhoaHoc\MODEL\inception_v4\incepV4-aug.pth')

# # # --- 2. Tiền xử lý ảnh đầu vào ---
# # def preprocess_image(img_path):
# #     transform = transforms.Compose([
# #         transforms.Resize((299, 299)),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # chuẩn hóa cho Inception
# #     ])
# #     img = Image.open(img_path).convert('RGB')
# #     return transform(img).unsqueeze(0)  # output: (1, 3, 299, 299)

# # # --- 3. Chuẩn hóa feature map về [0,1] để hiển thị ---
# # def normalize_fmap(fmap):
# #     fmap = fmap - fmap.min()
# #     fmap = fmap / (fmap.max() + 1e-5)
# #     return fmap

# # # --- 4. Vẽ lưới feature maps ---
# # def plot_feature_maps_grid_concat(fmaps_list, titles_list, max_cols=5, cmap='viridis'):
# #     num_maps = len(fmaps_list)
# #     if num_maps == 0:
# #         print("Warning: No feature maps to display.")
# #         return

# #     num_cols = min(max_cols, num_maps)
# #     num_rows = (num_maps + num_cols - 1) // num_cols

# #     plt.figure(figsize=(num_cols * 3, num_rows * 3))
# #     for i, fmap in enumerate(fmaps_list):
# #         ax = plt.subplot(num_rows, num_cols, i + 1)
# #         fmap_norm = normalize_fmap(fmap)
# #         plt.imshow(fmap_norm, cmap=cmap)
# #         plt.title(titles_list[i], fontsize=9)
# #         plt.axis('off')
# #     plt.tight_layout()
# #     plt.show()




# # import matplotlib.pyplot as plt
# # import matplotlib.gridspec as gridspec
# # import numpy as np
# # import tensorflow as tf
# # from tensorflow.keras.models import load_model

# # def normalize_feature_map(fmap):
# #     fmap -= np.min(fmap)
# #     fmap /= (np.max(fmap) + 1e-6)
# #     return fmap

# # def visualize_top2_per_block(img_path, feature_extractor, display_layer_names):
# #     img_raw = tf.io.read_file(img_path)
# #     img = tf.image.decode_image(img_raw, channels=3)
# #     img = tf.image.resize(img, (224, 224))
# #     img_tensor = tf.keras.applications.vgg16.preprocess_input(tf.expand_dims(img, axis=0))

# #     features = feature_extractor(img_tensor)
# #     feature_maps_np = [fmap.numpy()[0] for fmap in features]

# #     topk_feature_maps = []
# #     titles = []

# #     for i, fmap in enumerate(feature_maps_np):
# #         scores = np.sum(np.abs(fmap), axis=(0, 1))
# #         top2_idx = np.argsort(scores)[-2:][::-1]
# #         for ch in top2_idx:
# #             topk_feature_maps.append(fmap[:, :, ch])
# #             titles.append(f"{display_layer_names[i]}")

# #     num_maps = len(topk_feature_maps)
# #     num_cols = 5
# #     num_rows = (num_maps + num_cols - 1) // num_cols
# #     rows_for_img_col = max(num_rows, 3)

# #     plt.figure(figsize=(3 * (num_cols + 1), max(3, rows_for_img_col * 3)))
# #     gs = gridspec.GridSpec(nrows=rows_for_img_col, ncols=num_cols + 1, width_ratios=[1.5] + [1]*num_cols)

# #     middle_row = rows_for_img_col // 2
# #     ax_img = plt.subplot(gs[middle_row-1:middle_row+1, 0])
# #     ax_img.imshow(img.numpy().astype(np.uint8))
# #     ax_img.set_title("Ảnh gốc", fontsize=12)
# #     ax_img.axis('off')

# #     for i, fmap in enumerate(topk_feature_maps):
# #         row = i // num_cols
# #         col = i % num_cols + 1
# #         ax = plt.subplot(gs[row, col])
# #         ax.imshow(normalize_feature_map(fmap), cmap='viridis')
# #         ax.set_title(titles[i], fontsize=9)
# #         ax.axis('off')

# #     plt.tight_layout()
# #     plt.show()

# # from tensorflow.keras.applications import VGG16
# # from tensorflow.keras.models import Model

# # selected_layer_names = [
# #     'block1_conv2',
# #     'block2_conv2',
# #     'block3_conv3',
# #     'block4_conv3',
# #     'block5_conv3',
# # ]

# # model = load_model(r'E:\NghienCuuKhoaHoc\MODEL\vgg16\vgg16_aug_best_params_final.keras', compile=False)
# # outputs = [model.get_layer(name).output for name in selected_layer_names]
# # feature_extractor = Model(inputs=model.input, outputs=outputs)

# # visualize_top2_per_block(
# #     img_path=r"E:\LuanVan\data\raw\Maps\04-BE-VO NAM SON(24-38)004_page_26_img_1.png",
# #     feature_extractor=feature_extractor,
# #     display_layer_names=selected_layer_names
# # )



# # import torch
# # import timm
# # import torch.nn as nn
# # from torchvision import transforms
# # from PIL import Image
# # import matplotlib.pyplot as plt
# # import matplotlib.gridspec as gridspec
# # import numpy as np
# # import cv2

# # # Tiền xử lý ảnh
# # def preprocess_image(img_path):
# #     transform = transforms.Compose([
# #         transforms.Resize((299,299)),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
# #     ])
# #     img_pil = Image.open(img_path).convert('RGB')
# #     img_tensor = transform(img_pil).unsqueeze(0)
# #     return img_tensor, img_pil

# # # Chuẩn hóa feature map về 0-1
# # def normalize_feature_map(fmap):
# #     fmap = fmap - fmap.min()
# #     fmap = fmap / (fmap.max() + 1e-6)
# #     return fmap

# # # Lấy feature maps từ các layer cụ thể
# # class FeatureExtractor(nn.Module):
# #     def __init__(self, model, layer_names):
# #         super().__init__()
# #         self.model = model
# #         self.layer_names = layer_names
# #         self.features = {}
# #         self._register_hooks()

# #     def _register_hooks(self):
# #         def get_hook(name):
# #             def hook(module, input, output):
# #                 self.features[name] = output.detach()
# #             return hook

# #         # đăng ký hook với các layer
# #         for name in self.layer_names:
# #             layer = dict([*self.model.named_modules()]).get(name, None)
# #             if layer is not None:
# #                 layer.register_forward_hook(get_hook(name))
# #             else:
# #                 print(f"Không tìm thấy layer tên {name}")

# #     def forward(self, x):
# #         self.features = {}
# #         _ = self.model(x)
# #         return [self.features[name] for name in self.layer_names]

# # # Hiển thị ảnh gốc và các top2 feature map của từng layer
# # def visualize_top2_per_block(img_path, feature_extractor, display_layer_names):
# #     img_tensor, img_pil = preprocess_image(img_path)
# #     device = next(feature_extractor.model.parameters()).device
# #     img_tensor = img_tensor.to(device)

# #     feature_maps = feature_extractor(img_tensor)
# #     feature_maps_np = [fmap.cpu().numpy()[0] for fmap in feature_maps]  # (C,H,W)

# #     topk_feature_maps = []
# #     titles = []

# #     for i, fmap in enumerate(feature_maps_np):
# #         # Tính điểm cho từng channel
# #         scores = np.sum(np.abs(fmap), axis=(1, 2))
# #         top2_idx = np.argsort(scores)[-2:][::-1]
# #         for ch in top2_idx:
# #             topk_feature_maps.append(fmap[ch])
# #             titles.append(display_layer_names[i])

# #     num_maps = len(topk_feature_maps)
# #     num_cols = 5
# #     num_rows = (num_maps + num_cols - 1) // num_cols
# #     rows_for_img_col = max(num_rows, 3)

# #     plt.figure(figsize=(3 * (num_cols + 1), max(3, rows_for_img_col * 3)))
# #     gs = gridspec.GridSpec(nrows=rows_for_img_col, ncols=num_cols + 1, width_ratios=[1.5] + [1]*num_cols)

# #     # Vẽ ảnh gốc bên trái, chiếm 2 hàng
# #     middle_row = rows_for_img_col // 2
# #     ax_img = plt.subplot(gs[middle_row-1:middle_row+1, 0])
# #     ax_img.imshow(img_pil)
# #     ax_img.set_title("Ảnh gốc", fontsize=12)
# #     ax_img.axis('off')

# #     # Vẽ các feature map top2 của từng block bên phải
# #     for i, fmap in enumerate(topk_feature_maps):
# #         row = i // num_cols
# #         col = i % num_cols + 1
# #         ax = plt.subplot(gs[row, col])
# #         ax.imshow(normalize_feature_map(fmap), cmap='viridis')
# #         ax.set_title(titles[i], fontsize=9)
# #         ax.axis('off')

# #     plt.tight_layout()
# #     plt.show()

# # if __name__ == "__main__":
# #     # Load model InceptionV4 (hoặc model bạn đang dùng)
# #     model = timm.create_model('inception_v4', pretrained=True)
# #     model.eval()
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     model.to(device)

# #     # Tên các layer bạn muốn lấy feature map
# #     # Có thể thay bằng tên các layer trong model inception_v4, ví dụ:
# #     layer_names = [
# #         'features.0',  # stem block
# #         'features.1',  # inception_a
# #         'features.2',  # reduction_a
# #         'features.3',  # inception_b
# #         'features.4',  # reduction_b
# #         'features.5',  # inception_c
# #     ]
# #     display_layer_names = ['Conv', 'Inception A', 'Reduction A', 'Inception B', 'Reduction B', 'Inception C']

# #     extractor = FeatureExtractor(model, layer_names)

# #     image_path = r"E:\LuanVan\data\raw\Maps\04-BE-VO NAM SON(24-38)004_page_26_img_1.png"

# #     visualize_top2_per_block(image_path, extractor, display_layer_names)
