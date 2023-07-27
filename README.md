<p align="center"><img src="assets/hps_banner.png"/ width="100%"><br></p>

## HPS v2: Benchmarking Text-to-Image Generative Models

This is the official repository for the paper: [Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis](https://arxiv.org/abs/2306.09341). 

## Updates
*  [07/27/2023] We included `SDXL Base 0.9` model in the benchmark. It ranks 3rd on our benchmark!
*  [07/26/2023] We updated our [compressed checkpoint](https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt).
*  [07/19/2023] Live demo is available at 🤗[Hugging Face](https://huggingface.co/spaces/xswu/HPSv2).
*  [07/18/2023] We released our [test data](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155172150_link_cuhk_edu_hk/EUsElAcJO4FIkspfmSC5RbgBHL-kz85t5nwkM0waegq_bA?e=LH9Ret).

## Overview 
<p align="center"><img src="assets/overview.png"/ width="100%"><br></p>

**Human Preference Dataset v2 (HPD v2)**: a large-scale (798k preference choices / 430k images), a well-annotated dataset of human preference choices on images generated by text-to-image generative models. 

**Human Preference Score v2 (HPS v2)**: a preference prediction model trained on HPD v2. HPS v2 exhibits better correlation with human preferences against existing models. We also provide a fair, stable, and easy-to-use set of evaluation prompts for text-to-image generative models.

## The HPS v2 benchmark
The HPS v2 benchmark tracks a model's capability of generating images of 4 styles: *Animation*, *Concept-art*, *Painting*, and *Photo*. 

**The benchmark is actively updating, email us @ tgxs002@gmail.com or raise an issue if you feel your model/method needs to be included in this benchmark!**

| Model                 | Animation | Concept-art | Painting | Photo    | Averaged |
| ---------------------| --------- | ----------- | -------- | -------- | -------- |
| Dreamlike Photoreal 2.0 | 0.2824  | 0.2760      | 0.2759   | 0.2799   | 0.2786 |
| Realistic Vision      | 0.2822    | 0.2753      | 0.2756   | 0.2775   | 0.2777 |
| SDXL Base 0.9         | 0.2842    | 0.2763      | 0.2760   | 0.2729   | 0.2773 |
| Deliberate            | 0.2813    | 0.2746      | 0.2745   | 0.2762   | 0.2767 |
| ChilloutMix           | 0.2792    | 0.2729      | 0.2732   | 0.2761   | 0.2754 |
| MajicMix Realistic    | 0.2788    | 0.2719      | 0.2722   | 0.2764   | 0.2748 |
| Openjourney           | 0.2785    | 0.2718      | 0.2725   | 0.2753   | 0.2745 |
| DeepFloyd-XL          | 0.2764    | 0.2683      | 0.2686   | 0.2775   | 0.2727 |
| Epic Diffusion        | 0.2757    | 0.2696      | 0.2703   | 0.2749   | 0.2726 |
| Stable Diffusion v2.0 | 0.2748    | 0.2689      | 0.2686   | 0.2746   | 0.2717 |
| Stable Diffusion v1.4 | 0.2726    | 0.2661      | 0.2666   | 0.2727   | 0.2695 |
| DALL·E 2              | 0.2734    | 0.2654      | 0.2668   | 0.2724   | 0.2695 |
| Versatile Diffusion   | 0.2659    | 0.2628      | 0.2643   | 0.2705   | 0.2659 |
| CogView2              | 0.2650    | 0.2659      | 0.2633   | 0.2644   | 0.2647 |
| VQGAN + CLIP          | 0.2644    | 0.2653      | 0.2647   | 0.2612   | 0.2639 |
| DALL·E mini           | 0.2610    | 0.2556      | 0.2556   | 0.2612   | 0.2583 |
| Latent Diffusion      | 0.2573    | 0.2515      | 0.2525   | 0.2697   | 0.2578 |
| FuseDream             | 0.2526    | 0.2515      | 0.2513   | 0.2557   | 0.2528 |
| VQ-Diffusion          | 0.2497    | 0.2470      | 0.2501   | 0.2571   | 0.2510 |
| LAFITE                | 0.2463    | 0.2438      | 0.2443   | 0.2581   | 0.2481 |
| GLIDE                 | 0.2334    | 0.2308      | 0.2327   | 0.2450   | 0.2355 |

### Reproduce
We provide images for setting up the benchmark, and a script to reproduce it. Please see [Evaluation](#evaluation) for details.
The inference parameters and instructions for each model are provided in our paper, so you can also reproduce the table by your own images. 

## Human Preference Dataset v2
The prompts in our dataset are sourced from DiffusionDB and MSCOCO Captions. Prompts from DiffusionDB are first cleaned by ChatGPT to remove biased function words. Human annotators are tasked to rank images generated by different text-to-image generative models from the same prompt. Totally there are about 798k pairwise comparisons of images for over 430k images and 107k prompts, 645k pairs for training split and 153k pairs for test split.

Image sources of HPD v2:
|  Source | # of images 
| :-----: | :-----: |
| CogView2 | 73697 |
| DALL·E 2 | 101869 | 
| GLIDE (mini) | 400 |
| Stable Diffusion v1.4 | 101869 |
| Stable Diffusion v2.0 | 101869 | 
| LAFITE | 400 | 
| VQ-GAN+CLIP | 400 |
| VQ-Diffusion | 400 |
| FuseDream | 400 |
| COCO Captions | 28272 |

The dataset will be **released soon**.
Once unzipped, you should get a folder with the following structure:
```
HPD
---- train/
-------- {image_id}.jpg
---- test/
-------- {image_id}.jpg
---- train.json
---- test.json
---- benchmark/
-------- benchmark_imgs/
------------ {model_id}/
---------------- {image_id}.jpg
-------- drawbench/
------------ {model_id}/
---------------- {image_id}.jpg
-------- anime.json
-------- concept-art.json
-------- paintings.json
-------- photo.json
-------- drawbench.json
```

The annotation file, `train.json`, is organized as:
```
[
    {
        'human_preference': list[int], # 1 for preference
        'prompt': str,
        'file_path': list[str],
        'user_hash': str,
    },
    ...
]
```

The annotation file, `test.json`, is organized as:
```
[
    {
        'prompt': str,
        'image_path': list[str],
        'rank': list[int], # ranking for image at the same index in image_path
    },
    ...
]
```

The benchmark prompts file, ie. `anime.json` is pure prompts. The corresponding image can be found in the folder of the corresponding model by indexing the prompt.

## Environments

```
# environments
pip install -r requirements.txt 
```

## Evaluation

### Evaluating HPS v2

Evaluating HPS v2's correlation with human preference choices:
|  Model | Acc. on ImageReward test set (%)| Acc. on HPD v2 test set (%)
| :-----: | :-----: |:-----: |
|  [Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) | 57.4 | 72.6 |
|  [ImageReward](https://github.com/THUDM/ImageReward) | 65.1 | 70.6 |
|  [HPS](https://github.com/tgxs002/align_sd) | 61.2 | 73.1 |
|  [PickScore](https://github.com/yuvalkirstain/PickScore) | 62.9 | 79.8 |
|  Single Human | 65.3 | 78.1 |
|  HPS v2 | 65.7 | 83.3 |



HPS v2 checkpoint can be downloaded from [here](https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt). The model and live demo is also hosted on 🤗 Hugging Face at [here](https://huggingface.co/spaces/xswu/HPSv2).

Run the following commands to evaluate the HPS v2 model on HPD v2 test set and ImageReward test set:
```shell
# evaluate on HPD v2 test set
python evaluate.py --data-type test --data-path /path/to/HPD --image-path /path/to/image_folder --batch-size 10 --checkpoint /path/to/HPSv2.pt

# evaluate on ImageReward test set
python evaluate.py --data-type ImageReward --data-path /path/to/IR --image-path /path/to/image_folder --batch-size 10 --checkpoint /path/to/HPSv2.pt
```

### Evaluating text-to-image generative models using HPS v2
The generated images in our experiments can be downloaded from [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155172150_link_cuhk_edu_hk/EUsElAcJO4FIkspfmSC5RbgBHL-kz85t5nwkM0waegq_bA?e=LH9Ret). 
The following script reproduces the [benchmark table](#the-hps-v2-benchmark) and our results on DrawBench (reported in the paper):
```shell
# HPS v2 benchmark
python evaluate.py --data-type benchmark --data-path /path/to/HPD/benchmark --image-path /path/to/benchmark_imgs --batch-size 10 --checkpoint /path/to/HPSv2.pt

# DrawBench
python evaluate.py --data-type drawbench --data-path /path/to/HPD/benchmark --image-path /path/to/drawbench_imgs --batch-size 10 --checkpoint /path/to/HPSv2.pt
```

### Scoring single generated image and corresponding prompt

We provide one example image in the `asset/images` directory of this repo. The corresponding prompt is `"A cat with two horns on its head"`.

Run the following commands to score the single generated image and the corresponding prompt:
```shell
python score.py --image_path assets/demo_image.jpg --prompt 'A cat with two horns on its head'
```

## Train Human Preference Predictor
To train your own human preference predictor, just change the corresponding path in `configs/controller.sh` and run the following command:
```
# if you are running locally
bash configs/HPSv2.sh train ${GPUS} local
# if you are running on slurm
bash configs/HPSv2.sh train ${GPUS} ${quota_type}
```

## BibTeX
```
@article{wu2023human,
  title={Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis},
  author={Wu, Xiaoshi and Hao, Yiming and Sun, Keqiang and Chen, Yixiong and Zhu, Feng and Zhao, Rui and Li, Hongsheng},
  journal={arXiv preprint arXiv:2306.09341},
  year={2023}
}
```
