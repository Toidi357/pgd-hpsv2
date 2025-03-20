import torch
from PIL import Image
from typing import List


class PGD():
    """
    Modified from torchattacks.attacks.PGD source code
    """
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, device="cuda"):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device

    def forward(self, images, texts):
        r"""
        Overridden.
        """

        image_adv = images.clone().detach()
        image_adv.requires_grad = True
        
        # need to rescale our data based on the preprocessing from the image
        mean = torch.tensor(getattr(self.model.visual, "image_mean")).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(getattr(self.model.visual, "image_std")).view(1, 3, 1, 1).to(self.device)

        eps_norm = self.eps / std

        for i in range(self.steps):
            self.model.zero_grad()
            
            with torch.amp.autocast("cuda"):          
                # from train.py
                output = self.model(image_adv, texts)
                image_features, text_features, logit_scale = output["image_features"], output["text_features"], output["logit_scale"]
                logits_per_text = logit_scale * text_features @ image_features.T
                hps_score = torch.diagonal(logits_per_text)[0]
            
            # We want to minimize 'score'
            loss = hps_score  # since score is a scalar, treat it as the loss to minimize
            loss.backward()  # compute gradients dloss/dimage
            
            grad = image_adv.grad
            image_adv = image_adv - self.alpha * torch.sign(grad)
            
            delta = image_adv - images
            delta = torch.max(torch.min(delta, eps_norm), -eps_norm)
            image_adv = images + delta
            
            # need to account for the preprocessing done to the image
            min_norm = (0 - mean) / std
            max_norm = (1 - mean) / std
            image_adv = torch.max(torch.min(image_adv, max_norm), min_norm)
            
            image_adv = image_adv.detach()
            image_adv.requires_grad = True

        return image_adv
        
        
def preprocess_data(preprocess_val, tokenizer, device, file_path: str, prompt: str):
    """
    Taken from hpsv2/img_score.py
    """
    # Process the image
    image = preprocess_val(Image.open(file_path)).unsqueeze(0).to(device=device, non_blocking=True)
    # Process the prompt
    text = tokenizer([prompt]).to(device=device, non_blocking=True)
    return image, text


def calculateHPS(model, image, text) -> List[float]:
    with torch.no_grad():
        # Calculate the HPS
        with torch.amp.autocast("cuda"):          
            # from train.py
            output = model(image, text)
            image_features, text_features, logit_scale = output["image_features"], output["text_features"], output["logit_scale"]
            logits_per_text = logit_scale * text_features @ image_features.T
            hps_score = torch.diagonal(logits_per_text).cpu().numpy()
            
        return hps_score
        

# chatgpt function that takes the preprocessed image and takes it back to a normal image
def denormalize(image, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
    """
    Reverses the CLIP preprocessing normalization.
    
    Args:
        image (torch.Tensor): A normalized image tensor of shape (C, H, W) or (N, C, H, W).
        mean (tuple): The mean values used for normalization.
        std (tuple): The std values used for normalization.
    
    Returns:
        torch.Tensor: The denormalized image with pixel values in [0, 1].
    """
    if image.dim() == 4:
        # For a batch of images
        mean_tensor = torch.tensor(mean).view(1, 3, 1, 1).to(image.device)
        std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(image.device)
    else:
        # For a single image
        mean_tensor = torch.tensor(mean).view(3, 1, 1).to(image.device)
        std_tensor = torch.tensor(std).view(3, 1, 1).to(image.device)
    
    # Reverse the normalization: x = x_norm * std + mean
    image = image * std_tensor + mean_tensor
    # Optionally, clip to [0, 1] if needed.
    image = torch.clamp(image, 0, 1)
    return image