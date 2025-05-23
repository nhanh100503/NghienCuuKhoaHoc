import torch
from torchvision import models
import timm
import torch.optim as optim

import torch.nn as nn
import torch
from torch.optim import AdamW

def build_convnextv2_raw_model(num_classes, lr=0.000541, dropout_rate=0.112936, dense_units=256,
                l2_reg=0.000081, fine_tune_all=False, unfreeze_from_stage=2):

    model = timm.create_model('convnextv2_tiny.fcmae_ft_in1k', pretrained=True, num_classes=0)
    in_features = model.head.in_features

    # Đóng băng toàn bộ nếu không fine-tune tất cả
    for param in model.parameters():
        param.requires_grad = False

    # Mở các stage từ vị trí chỉ định
    for stage in model.stages[unfreeze_from_stage:]:
        for param in stage.parameters():
            param.requires_grad = True

    # Head mới với các tham số từ trial
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(in_features, dense_units),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(dense_units, num_classes)
    )

    return model



def build_convnext_v2_aug_model(num_classes, lr=0.000466, 
                l2_reg=0.000017  , fine_tune_all=False, unfreeze_from_stage=3):

    # Tải mô hình ConvNeXt v2 với pretrained weights
    model = timm.create_model('convnextv2_tiny.fcmae_ft_in1k', pretrained=True, num_classes=0)
    in_features = model.head.in_features

    # Đóng băng toàn bộ nếu không fine-tune tất cả
    for param in model.parameters():
        param.requires_grad = False

    # Mở các stage từ vị trí chỉ định (unfreeze từ stage 2 trở đi)
    for i, stage in enumerate(model.stages):
        if i >= unfreeze_from_stage:
            for param in stage.parameters():
                param.requires_grad = True

    # Head mới với các tham số từ trial
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.GELU(),  
        nn.Linear(in_features, 11) # Linear classifier
    )

    return model



def load_pytorch_model(model_name, weight_path, num_classes):
    if model_name == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    else:
        raise ValueError("Only alexnet and convnextv2 supported here.")

    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model


def load_convnextv2_model(model_path, num_classes, model_type):
    if model_type == "convnext_v2_raw":
        model = build_convnextv2_raw_model(num_classes=num_classes)
    else:
        model = build_convnext_v2_aug_model(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    # print(checkpoint['model_state_dict'].keys())

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    return model



#     return model
def load_alexnet_model(path, num_classes, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = models.AlexNet_Weights.IMAGENET1K_V1
    model = models.alexnet(weights=weights)

    # Thay đổi classifier theo số lớp
    model.classifier = nn.Sequential(
        nn.Linear(256 * 6 * 6, 256),
        nn.ReLU(),
        nn.Dropout(p=0.342544),
        nn.Linear(256, num_classes)
    )

    # Freeze tất cả layers trước layer thứ 4 (tính từ 0)
    for param in model.features.parameters():
        param.requires_grad = False

    # Mở khóa layers từ thứ 4 (index >= 3)
    for i in range(3, len(model.features)):
        for param in model.features[i].parameters():
            param.requires_grad = True

    # Luôn train classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.000096,
        weight_decay=0.000678
    )

    criterion = nn.CrossEntropyLoss()

    # Load checkpoint và lấy đúng phần model_state_dict 
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)  # Trong trường hợp chỉ chứa state_dict

    model.to(device)
    model.eval()

    return model
  