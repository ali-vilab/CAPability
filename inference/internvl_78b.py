import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import os
import math
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import datetime
import argparse
# from utils import CUDADataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--complex_prompt", action='store_true', default=False)

    return parser.parse_args()


class EvalDataset(Dataset):
    def __init__(self, data_list, complex_prompt=False):
        super(EvalDataset, self).__init__()
        if complex_prompt:
            self.image_prompt = "Please describe the image in detail. Your description should follow these rules:\n"\
                "a) You should describe each object in the image in detail, including its name, number, color, and spatial relationship between objects.\n"\
                "b) You should describe the scene of the image.\n"\
                "c) You should describe the camera angle when shooting this image, such as level angle, high angle, low angle, or dutch angle.\n"\
                "d) You should describe the style of the image, such as realistic, animated, special-effect, old-fashioned and so on.\n"\
                "e) If there are any texts in the image, you should describe the text content.\n"\
                "f) If you know the character in the image, you should tell his or her name.\n"\
                "Directly output your detailed description in a elaborate paragraph, instead of itemizing them in list form. Your description: "
            self.video_prompt = "Please describe the video in detail. Your description should follow these rules:\n"\
                "a) You should describe each events in the video in order, especially focusing on the behavior and action of characters, including people, animals.\n"\
                "b) You should describe each object in the video in detail, including its name, number, color, and spatial relationship between objects.\n"\
                "c) You should describe the scene of the video.\n"\
                "d) You should describe the camera movement when shooting this video, especially the direction, such as pan left, track right, tilt up, boom down, zoom in, dolly out, and so on.\n"\
                "e) You should describe the style of the video, such as realistic, animated, special-effect, old-fashioned and so on.\n"\
                "f) If there are any texts in the video, you should describe the text content.\n"\
                "g) If you know the character in the video, you should tell his or her name.\n"\
                "Directly output your detailed description in a elaborate paragraph, instead of itemizing them in list form. Your description: "
        else:
            self.image_prompt = "Please describe the image in detail."
            self.video_prompt = "Please describe the video in detail."
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        case = self.data_list[index]
        if case['data_type'] == 'image':
            data_type = 'image'
            file_path = "../" + case["file_path"]
            query = f"<image>\n{self.image_prompt}"
            pixel_values = load_image(file_path, max_num=12).to(torch.bfloat16)
            num_patches_list = None
        else:
            data_type = 'video'
            file_path = "../" + case["file_path"]
            query = self.video_prompt
            pixel_values, num_patches_list = load_video(file_path, num_segments=8, max_num=1)
            pixel_values = pixel_values.to(torch.bfloat16)
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            query = video_prefix + query
        return case, data_type, query, pixel_values, num_patches_list

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def main():
    args = parse_args()
    ann = args.benchmark + '.jsonl'
    with open(f"../annotations/{ann}", 'r') as f:
        meta = f.readlines()
    meta = [json.loads(i) for i in meta]
    os.makedirs(args.save_root, exist_ok=True)
    if os.path.exists(f"{args.save_root}/{ann}"):
        with open(f"{args.save_root}/{ann}", 'r') as f:
            processed_data = [json.loads(l.strip('\n')) for l in f.readlines()]
        processed_data = [d['file_id'] for d in processed_data]
        meta = [d for d in meta if d['file_id'] not in processed_data]
        print(f"Find {len(processed_data)} records, remaining {len(meta)} to be processed")
        if len(meta) == 0:
            return

    # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    device_map = split_model('InternVL2_5-78B') # if 'InternVL2_5-78B' in args.model_path else {"": f"cuda:{local_rank}"}
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    dataset = EvalDataset(meta, complex_prompt=args.complex_prompt)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        shuffle=True,
        collate_fn=lambda x: x[0],  # asume the batch_size is always 1
        pin_memory=True,
    )

    generation_config = dict(max_new_tokens=8192, do_sample=False)
    results = []
    for n, (case, data_type, query, pixel_values, num_patches_list) in enumerate(tqdm(dataloader, desc=args.benchmark, total=len(dataloader))):
        if data_type == "image":
            response = model.chat(tokenizer, pixel_values, query, generation_config=generation_config)
        else:
            response = model.chat(tokenizer, pixel_values, query, generation_config,
                                        num_patches_list=num_patches_list)

        
        with open(f"{args.save_root}/{ann}", 'a', encoding='utf-8') as f:
            f.write(json.dumps({'file_id': case['file_id'], 'caption': response}, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
