# analysis/interpretability.py

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


def show_gradcam_on_image(img_tensor, cam, title="Grad-CAM"):
    """
    img_tensor: [C,H,W], torch.Tensor
    cam: numpy heatmap [H,W]
    """
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    plt.imshow(img)
    plt.imshow(cam, cmap="jet", alpha=0.5)
    plt.title(title)
    plt.axis("off")
    plt.show()


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


def show_saliency_on_image(img_tensor, saliency_map, title="Saliency Map"):
    """
    img_tensor: [C,H,W], torch.Tensor
    saliency_map: numpy heatmap [H,W]
    """
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    plt.imshow(img)
    plt.imshow(saliency_map, cmap="hot", alpha=0.5)
    plt.title(title)
    plt.axis("off")
    plt.show()
