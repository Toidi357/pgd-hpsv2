import torchvision.transforms as transforms
import torch
import warnings
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import attack_utils
from typing import List, Dict, Tuple
import numpy as np
import os
import json
from pprint import pprint
import time
from tqdm import tqdm


def initialize_model():
    """
    This function taken from hpsv2/img_score.py
    """
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val
        

def get_first_k_images(k: int = 10) -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns the first k images from each benchmark located in datasets/benchmark/benchmark_imgs
    
    Args:
        k: integer
    
    Returns:
        Dict[str, List[Tuple[str, str]]]: A dictionary mapping of a benchmark name to a list of (prompt, image_path) tuples
    """
    # load the prompts from anime.json  
    with open(os.path.join('.', 'datasets/benchmark/anime.json'), 'r') as f:
        prompts = json.loads(f.read())[:k]
    
    _ = {}
    
    base_path = 'datasets/benchmark/benchmark_imgs'
    for dir in os.listdir(os.path.join('.', base_path)):
        full_path = os.path.join(base_path, dir)
    
        if os.path.isdir(full_path):
            print(f"Processing directory: {dir}")
            
            # the images are like 00000.jpg, 00001.jpg, etc
            image_paths = [os.path.join(full_path, "anime/" + str(i).zfill(5) + ".jpg") for i in range(k)]
            
            _[dir] = [(prompts[i], image_paths[i]) for i in range(len(image_paths))]
    
    return _


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    start_time = time.time()

    pprint("Initializing model...")
    model_dict = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    initialize_model()
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']
    checkpoint = torch.load("HPS_v2_compressed.pt", map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    print(f"Finished initializing model in {time.time() - start_time} seconds")
    
    # load images from the downloaded benchmark dataset
    images = get_first_k_images(10)
    
    # initialize our PGD attack class
    attack = attack_utils.PGD(model, eps=8/(255*1), alpha=1/(255*1), steps=2)
    
    # used for writing images to disk
    to_pil = transforms.ToPILImage()
    
    scores = {}
    for benchmark_name, data in images.items():
        pprint(f"Performing experiments for benchmark {benchmark_name}...")
        
        # create the output directory for pgd'ed images
        os.makedirs(os.path.dirname(os.path.join('.', f'adv_images/{benchmark_name}/')), exist_ok=True)
        
        original_scores = []
        pgd_scores = []
        for prompt, image_path in tqdm(data):
            # preprocess the image
            image, text = attack_utils.preprocess_data(preprocess_val, tokenizer, device, image_path, prompt)
            
            # calculate its original HPS score
            hps_score = attack_utils.calculateHPS(model, image, text)[0]
            
            # perform PGD to get an adversarial image
            adv_image = attack.forward(image, text)
            
            # calculate PGD'ed HPS score
            adv_hps_score = attack_utils.calculateHPS(model, adv_image, text)[0]
            
            # save everything and write to disk
            original_scores.append(float(hps_score))
            pgd_scores.append(float(adv_hps_score))

            # have to de-preprocess the image to write to disk
            denorm_image = attack_utils.denormalize(image, mean=getattr(model.visual, "image_mean"), std=getattr(model.visual, "image_std"))
            img_pil = to_pil(denorm_image.squeeze().cpu().detach())
            img_pil.save(os.path.join('.', f'adv_images/{benchmark_name}/orig_{os.path.basename(image_path)}'))
            
            denorm_adv_image = attack_utils.denormalize(adv_image, mean=getattr(model.visual, "image_mean"), std=getattr(model.visual, "image_std"))
            img_pil = to_pil(denorm_adv_image.squeeze().cpu().detach())
            img_pil.save(os.path.join('.', f'adv_images/{benchmark_name}/adv_{os.path.basename(image_path)}'))
            
        
        scores[benchmark_name] = {
            "avg_orig_hps_score": np.mean(original_scores),
            "orig_hps_scores": original_scores,
            "avg_adv_hps_score": np.mean(pgd_scores),
            "pgd_hps_scores": pgd_scores,
        }
        
        break
    
    pprint(scores)

    # write to disk
    with open('results.json', 'w') as f:
        f.write(json.dumps(scores))
