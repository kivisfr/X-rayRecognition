# analysis/interpretability.py
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# ============================
# Grad-CAM
# ============================

class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: trained model (torch.nn.Module)
        target_layer: layer on which to calculate activations (e.g., model.layer4[-1] for ResNet)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # hook для градиентов
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # hook для активаций
        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: image batch [B,C,H,W]
        target_class: class index (torch.Tensor or int)
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)

        # if target_class is int, convert it to a tensor
        if isinstance(target_class, int):
            target_class = torch.tensor([target_class], device=input_tensor.device)

        loss = F.cross_entropy(output, target_class)
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        # average of gradient channels
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # normalization
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


# ============================
# Saliency Maps
# ============================

def compute_saliency(model, input_tensor, target_class=None):
    """
    model: trained model
    input_tensor: [B,C,H,W]
    target_class: class index
    """
    model.eval()
    input_tensor.requires_grad_()

    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1)

    if isinstance(target_class, int):
        target_class = torch.tensor([target_class], device=input_tensor.device)

    loss = F.cross_entropy(output, target_class)
    model.zero_grad()
    loss.backward()

    saliency = input_tensor.grad.abs().max(dim=1)[0]  # [B,H,W]
    saliency = saliency.detach().cpu().numpy()
    return saliency


def save_saliency_on_image(img_tensor, saliency, out_path, title=None):
    """
    Сохраняет saliency map поверх изображения в файл.

    Parameters
    ----------
    img_tensor : torch.Tensor
        Изображение в формате (C, H, W).
    saliency : np.ndarray
        Карта чувствительности (H, W).
    out_path : str or Path
        Путь для сохранения итогового изображения.
    title : str, optional
        Заголовок для картинки (будет добавлен сверху).
    """
    out_path = Path(out_path)

    # преобразуем тензор в numpy
    img = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(saliency, cmap="jet", alpha=0.5)
    if title:
        plt.title(title)
    plt.axis("off")

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_gradcam_on_image(img_tensor, cam, out_path, title=None):
    """
    Сохраняет Grad-CAM карту поверх изображения в файл без отображения.

    Parameters
    ----------
    img_tensor : torch.Tensor
        Изображение в формате (C, H, W).
    cam : np.ndarray
        Grad-CAM карта (H, W).
    out_path : str or Path
        Путь для сохранения итогового изображения.
    title : str, optional
        Заголовок для картинки.
    """
    out_path = Path(out_path)

    # преобразуем тензор в numpy
    img = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())  # нормализация

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(cam, cmap="jet", alpha=0.5)  # накладываем Grad-CAM
    if title:
        plt.title(title)
    plt.axis("off")

    # сохраняем в файл
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()  # закрываем figure, чтобы не было отображения
