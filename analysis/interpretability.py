# analysis/interpretability.py
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch import nn

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
            target_class = torch.tensor([target_class], device=output.device)

        # If target_class is a tensor, we convert it to the correct type and device
        target_class = target_class.to(output.device).long()

        # If it's a tensor, make sure it has the shape [batch_size]
        if target_class.dim() == 0:
            target_class = target_class.unsqueeze(0)

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


def get_last_conv_layer(model: nn.Module):
    """
    Returns the last convolutional layer (Conv2d) in the model.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Conv2d layer not found in model")
    return last_conv


def save_saliency_on_image(img_tensor, saliency, out_path, title=None):
    """
     Saves a saliency map over an image to a file.

    Parameters
    ----------
    img_tensor : torch.Tensor
        Image in format (C, H, W).
    saliency : np.ndarray
        Sensitivity map (H, W).
    out_path : str or Path
        Path to save the final image.
    title : str, optional
        Title for the image (will be added at the top).
    """
    out_path = Path(out_path)

    # Convert a tensor to Numpy
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


def save_gradcam_on_image(img_tensor, cam, out_path, title=None):
    """
    Saves the Grad-CAM map over the image to a file without displaying it.

    Parameters
    ----------
    img_tensor : torch.Tensor
        Image in format (C, H, W).
    cam : np.ndarray
        Grad-CAM map (H, W).
    out_path : str or Path
        Path to save the final image.
    title : str, optional
        Title for the picture.
    """
    out_path = Path(out_path)

    # Convert a tensor to Numpy
    img = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())  # normalization

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(cam, cmap="jet", alpha=0.5)  # we apply Grad-CAM
    if title:
        plt.title(title)
    plt.axis("off")

    # сохраняем в файл
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# ---------- Основная функция ----------

def save_random_misclassified_examples(dataset, targets, preds, classes,
                                       model, model_name, split, out_dir,
                                       n_samples=2, device = "cuda"):
    """
    For each pair (true_label_as_pred_label), it saves random n_samples of images with Saliency and Grad-CAM visualization.
    After processing each pair, it clears the memory.
    """
    out_dir = Path(out_dir)
    pairs = {}

    # We collect indexes of erroneous examples
    for idx in range(len(dataset)):
        true, pred = targets[idx], preds[idx]
        if true != pred:
            key = (classes[true], classes[pred])
            pairs.setdefault(key, []).append(idx)

    target_layer = get_last_conv_layer(model)

    for (true_label, pred_label), indices in pairs.items():
        pair_dir = out_dir / model_name / split / f"{true_label}_as_{pred_label}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        chosen = random.sample(indices, min(n_samples, len(indices)))

        for idx in chosen:

            # --- correct access to the image ---
            if hasattr(dataset, "indices"):
                real_idx = dataset.indices[idx]
                img_tensor, _ = dataset.dataset[real_idx]
            else:
                img_tensor, _ = dataset[idx]

            img_tensor = img_tensor.unsqueeze(0).to(device)


            # --- Saliency ---
            img_tensor.requires_grad_()
            output = model(img_tensor)
            target_class = preds[idx]

            target_tensor = torch.tensor([target_class], device=output.device).long()
            loss = F.cross_entropy(output, target_tensor)
            model.zero_grad()
            loss.backward()

            saliency = img_tensor.grad.data.abs().squeeze().max(dim=0)[0].cpu().numpy()
            save_saliency_on_image(img_tensor.squeeze(), saliency,
                                   pair_dir / f"saliency_{idx}.png",
                                   title=f"Saliency {true_label}_as_{pred_label}")

            # --- Grad-CAM ---
            gradcam = GradCAM(model, target_layer)
            cam = gradcam.generate(img_tensor, target_class=target_class)
            save_gradcam_on_image(img_tensor.squeeze(), cam,
                                  pair_dir / f"gradcam_{idx}.png",
                                  title=f"Grad-CAM {true_label}_as_{pred_label}")

            # free up memory for each example
            del img_tensor, output, loss, saliency, cam, gradcam, target_tensor
            torch.cuda.empty_cache()

        # additional cleaning after processing the entire pair
        torch.cuda.empty_cache()