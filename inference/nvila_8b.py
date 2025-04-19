import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import datetime
import argparse
# import ffmpeg
import numpy as np
from decord import VideoReader, cpu
import llava
from llava import conversation as clib
from llava.media import Image, Video
from utils import CUDADataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--complex_prompt", action='store_true', default=False)

    return parser.parse_args()


class EvalDataset(Dataset):
    def __init__(self, data_list, num_splits, split_idx, complex_prompt=False):
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
        self.data_list = data_list[split_idx::num_splits]

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        case = self.data_list[index]
        if case['data_type'] == 'image':
            file_path = "../" + case["file_path"]
            query = self.image_prompt
            media = [Image(file_path)]
            media.append(query)
        else:
            file_path = "../" + case["file_path"]
            query = self.video_prompt
            media = [Video(file_path)]
            media.append(query)
        return case, media


def main():
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
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

    model = llava.load(args.model_path,
        device_map={"": f"cuda:{local_rank}"},
        attn_implementation="flash_attention_2",)

    dataset = EvalDataset(meta, num_splits=dist.get_world_size(), split_idx=global_rank, complex_prompt=args.complex_prompt)
    dataloader = CUDADataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        shuffle=True,
        collate_fn=lambda x: x[0],  # asume the batch_size is always 1
        pin_memory=True,
    )
    results = []
    generation_config = model.default_generation_config
    generation_config.update(**dict(max_new_tokens=8192, do_sample=False, temperature=0))
    for n, (case, media) in enumerate(tqdm(dataloader, desc=f"Rank {global_rank}", total=len(dataloader), position=local_rank)):

        response = model.generate_content(media, generation_config=generation_config)
        
        with open(f"{args.save_root}/{ann}", 'a', encoding='utf-8') as f:
            f.write(json.dumps({'file_id': case['file_id'], 'caption': response}, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
