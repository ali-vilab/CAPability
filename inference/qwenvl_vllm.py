import json
import math
import time
import base64
import random
import argparse
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
# from vision_process import process_vision_info
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def init_model(model_path, tensor_parallel_size):
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 1, "video": 1},
        tensor_parallel_size=tensor_parallel_size
    )
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=2048,
        stop_token_ids=[],
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return llm, sampling_params, processor


class QwenDataset(Dataset):
    def __init__(self, data_list, data_root, processor, video_kwargs=None, complex_prompt=False):
        super(QwenDataset, self).__init__()
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
        self.data_root = data_root
        self.processor = processor
        self.video_kwargs = video_kwargs

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data = self.data_list[index]
        llm_input = self.get_llm_input_from_data(data, self.video_kwargs)
        return data, llm_input

    def get_llm_input_from_data(self, data, video_kwargs=None):
        if data['data_type'] == 'image':
            file_path = data['file_path']
            text_prompt = self.image_prompt
            content = [
                {
                    "type": "image",
                    "image": f"file://{os.path.join(self.data_root, file_path)}",
                },
                {
                    "type": "text",
                    "text": text_prompt
                },
            ]
        elif data['data_type'] == 'video':
            file_path = data['file_path']
            text_prompt = self.video_prompt
            content = [
                {
                    "type": "video",
                    "video": f"file://{os.path.join(self.data_root, file_path)}",
                    # "fps": 2.0,
                    # "max_frames": 768,
                },
                {"type": "text", "text": text_prompt},
            ]
            if video_kwargs is not None:
                content[0].update(video_kwargs)
        else:
            raise ValueError("No image or video path in data")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": content,
            },
        ]
        
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_input = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        return llm_input

@dataclass
class DataCollator(object):
    def __call__(self, batch):
        datas = []
        llm_inputs = []
        for e in batch:
            datas.append(e[0])
            llm_inputs.append(e[1])
        return datas, llm_inputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=False, default=None)
    parser.add_argument("--error_save_root", type=str, required=False, default=None)
    parser.add_argument("--is_processing_large_videos", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--complex_prompt", action='store_true', default=False)
    return parser.parse_args()


def llm_infer(llm, llm_inputs, sampling_params):
    try:
        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
        generated_texts = [outputs[j].outputs[0].text for j in range(len(llm_inputs))]
    except Exception as e:
        print("Error:", e)
        return None
    
    return generated_texts

if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_path
    model_name = os.path.basename(model_path)
    tensor_parallel_size = args.num_gpus
    llm, sampling_params, processor = init_model(model_path, tensor_parallel_size)

    anno_root = "../annotations"
    data_root = "../"
    anno_files = [os.path.join(anno_root, filename) for filename in os.listdir(anno_root)]
    if args.benchmark is not None:
        anno_files = [f for f in anno_files if args.benchmark in f]
        assert len(anno_files) == 1
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)

    if args.error_save_root is not None:
        os.makedirs(args.error_save_root, exist_ok=True)

    for anno_file in tqdm(anno_files, desc="task"):
        if 'action' in anno_file:
            continue

        task_name = os.path.basename(anno_file).split('.')[0]
        save_file = os.path.join(save_root, f"{task_name}.jsonl")
        if args.error_save_root is not None:
            error_file = os.path.join(args.error_save_root, f"{task_name}.jsonl")
        with open(anno_file, 'r', encoding='utf-8') as f:
            data = [json.loads(l.strip('\n')) for l in f.readlines()]
        
        data_type = data[0]['data_type']
        if data_type == 'image':
            data_type = 'image'
            batch_size = 4 if '72B' in model_path else 24
        elif data_type == 'video':
            data_type = 'video'
            batch_size = 2 if '72B' in model_path else 4
        else:
            raise ValueError("No image or video path in data")
        
        

        print(f"[{task_name}] data length: {len(data)}")
        if os.path.exists(save_file):
            with open(save_file, 'r', encoding='utf-8') as f:
                processed_data = [json.loads(l.strip('\n')) for l in f.readlines()]
            processed_data = [d['file_id'] for d in processed_data]
            remain_data = [d for d in data if d['file_id'] not in processed_data]
            print(f"[{task_name}] Find {len(processed_data)} records, remaining {len(remain_data)} to be processed")
        else:
            remain_data = data
        if len(remain_data) == 0:
            continue
        
        if args.is_processing_large_videos and 'Qwen2.5' in args.model_path:
            video_kwargs = {
                "max_pixels": 360 * 512,
            }
            batch_size = 1
        elif args.is_processing_large_videos:
            video_kwargs = {
                "max_pixels": 360 * 360,
                "fps": 1.0,
                "max_frames": 384,
            }

        else:
            video_kwargs = None
        dataset = QwenDataset(remain_data, data_root, processor, video_kwargs, complex_prompt=args.complex_prompt)
        dataloader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=False, num_workers=8,
            collate_fn=DataCollator(), pin_memory=True
        )

        for i, (datas, llm_inputs) in enumerate(tqdm(dataloader, desc=task_name)):
            generated_texts = llm_infer(llm, llm_inputs, sampling_params)
            if generated_texts is None:
                assert data_type == "video"
                for d in datas:
                    llm_inputs = [dataset.get_llm_input_from_data(d)]
                    generated_texts = llm_infer(llm, llm_inputs, sampling_params)
                    if generated_texts is None:
                        if args.error_save_root is not None:
                            with open(error_file, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(d, ensure_ascii=False) + '\n')
                            continue
                    else:
                        with open(save_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps({'file_id': d['file_id'], 'caption': generated_texts[0]}, ensure_ascii=False) + '\n')
            else:
                with open(save_file, 'a', encoding='utf-8') as f:
                    for k,v in zip(datas, generated_texts):
                        f.write(json.dumps({'file_id':k['file_id'], 'caption': v}, ensure_ascii=False) + '\n')
            
        
        if args.error_save_root is not None and os.path.exists(error_file):
            with open(error_file, 'r', encoding='utf-8') as f:
                error_data = [json.loads(l.strip('\n')) for l in f.readlines()]
            if len(error_data) == 0:
                os.remove(error_file)