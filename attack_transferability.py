import torch
import clip
from PIL import Image
from typing import List, Dict, Tuple
import sys
import os
import warnings
import time
from pprint import pprint
from tqdm import tqdm
import numpy as np

def get_adversarial_images() -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Returns:
        Dict[str, List[Tuple[str, str, str]]]: A dictionary mapping of a benchmark name to a list of (prompt, orig_img, adv_img) tuples
    """
    # prompts are hard coded here b/c I didn't store them in adv_images
    prompts = [
        "Spongebob depicted in the style of Dragon Ball Z.",
        "Lionel Messi portrayed as a sitcom character.",
        "A digital artwork depicting a cartoon illustration of a warehouse environment.",
        "Two young Japanese goth cosplay girls in fishnets, corsets, chokers, and black and white makeup with full body tattoos and intricate painted details.",
        "A white-haired girl in a pink sweater looks out a window in her bedroom.",
        "A girl gazes at a city from a mountain at night in a colored manga illustration by Diego Facio.",
        "A hamster resembling a horse.",
        "The president being abducted by aliens.",
        "Anime-style fighter pilot in cockpit engaged in a night air battle with explosions.",
        "A hyper-realistic representation of the hypnotoad from Futurama.",
    ]
    N = len(prompts)
    
    _ = {}
    
    base_path = 'adv_images'
    for dir in os.listdir(os.path.join('.', base_path)):
        full_path = os.path.join(base_path, dir)
    
        if os.path.isdir(full_path):
            print(f"Processing directory: {dir}")
            
            # the images are like orig_00000.jpg, orig_00001.jpg, etc
            orig_img_paths = [os.path.join(full_path, "orig_" + str(i).zfill(5) + ".jpg") for i in range(N)]
            adv_img_paths = [os.path.join(full_path, "adv_" + str(i).zfill(5) + ".jpg") for i in range(N)]
            
            _[dir] = [(prompts[i], orig_img_paths[i], adv_img_paths[i]) for i in range(N)]
    
    return _

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    start_time = time.time()
    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    pprint("Initializing model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    params = torch.load("hpc.pt", weights_only=False)['state_dict']
    model.load_state_dict(params)
    model = model.to(device)
    print(f"Finished initializing model in {time.time() - start_time} seconds")

    # load images from adv_images
    images = get_adversarial_images()

    scores = {}
    for benchmark_name, data in images.items():
        pprint(f"Performing experiments for benchmark {benchmark_name}...")
        
        original_scores = []
        adv_scores = []
        for prompt, orig_img_path, adv_img_path in tqdm(data):
            # preprocess
            orig_image = preprocess(Image.open(orig_img_path)).unsqueeze(0).to(device)
            adv_image = preprocess(Image.open(adv_img_path)).unsqueeze(0).to(device)
            text = clip.tokenize([prompt]).to(device)
            
            # calculate unperturbed HPS
            with torch.no_grad():
                image_features = model.encode_image(orig_image)
                text_features = model.encode_text(text)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                hps = image_features @ text_features.T
                original_scores.append(float(hps[0]) * 100)
            
            # calculate perturbed HPS
            with torch.no_grad():
                image_features = model.encode_image(adv_image)
                text_features = model.encode_text(text)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                hps = image_features @ text_features.T
                adv_scores.append(float(hps[0]) * 100)
                
        scores[benchmark_name] = {
            "avg_orig_hps_score": np.mean(original_scores),
            "avg_adv_hps_score": np.mean(adv_scores),
        }
    
    pprint(scores)