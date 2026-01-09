# src/engine/demo_gradio_engine.py

import os

import yaml
import torch
import gradio as gr
from src.models import build_model
from src.datasets import CIFAR10Transforms

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model_and_transform(cfg, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} not found")

    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    class_names = ckpt["class_names"]
    transform = CIFAR10Transforms.test()
    num_classes = len(class_names)

    return model, transform, class_names, num_classes

def classify_image(image, model, transform, class_names, num_classes, device):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1)[0].cpu().tolist()

    return {class_names[i]: float(probs[i]) for i in range(num_classes)}

def build_demo(model, transform, class_names, num_classes, device):
    def predict_fn(image):
        return classify_image(
                image = image,
                model = model,
                transform = transform,
                class_names = class_names,
                num_classes = num_classes,
                device = device,
                )

    demo = gr.Interface(
            fn = predict_fn,
            inputs = gr.Image(type="pil", label="上传一张图片"),
            outputs = gr.Label(num_top_classes=5, label="预测结果(Top-5)"),
            title = "CIFAR图像分类",
            description = "上传一张图片,查看模型的预测结果",
            allow_flagging = "never",
            )

    return demo

def run_gradio(cfg_path):
    cfg = load_cfg(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device:{device}")

    ckpt_path = cfg["test"]["ckpt"]

    model, transform, class_names, num_classes = load_model_and_transform(
            cfg = cfg,
            ckpt_path = ckpt_path,
            device = device,
            )

    demo = build_demo(
            model = model,
            transform = transform,
            class_names = class_names,
            num_classes = num_classes,
            device = device,
            )
    
    demo.launch()
