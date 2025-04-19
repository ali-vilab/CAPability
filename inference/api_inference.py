import os
import oss2
import json
import time
import math
import uuid
import base64
import random
import argparse
import requests
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from openai import OpenAI
from contextlib import nullcontext
from multiprocessing import Pool, Lock, get_context
from oss2.credentials import EnvironmentVariableCredentialsProvider

from constant import *


OSS_URL = os.environ['OSS_URL']
OSS_BUCKET = os.environ['BUCKET']
auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
bucket = oss2.Bucket(auth, OSS_URL, OSS_BUCKET)

OPENAI_SERVER_URL = os.environ['OPENAI_SERVER_URL']
OPENAI_KEY = os.environ['OPENAI_KEY']

def url_to_base64(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        encoded_string = base64.b64encode(response.content).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"
    else:
        print(f"Error: {response.status_code}")
        raise ValueError(f"URL2BASE64 Error: {response.status_code}")
        return None


def get_content(model_name, data_type, data_url, user_prompt, max_frames=-1, to_base64=False):
    if 'gemini' in model_name:
        content = [
            {"type": f"{data_type}_url", f"{data_type}_url": {"url": data_url}},
            {"type": "text", "text": user_prompt}
        ]
    elif 'gpt' in model_name or 'claude' in model_name:
        if data_type == "video":
            assert isinstance(data_url, list)
            num_frames = len(data_url)
            if max_frames > 0 and num_frames > max_frames:
                frame_idx = np.linspace(0, num_frames-1, max_frames, dtype=int).tolist()
                url_list = [data_url[i] for i in frame_idx]
            else:
                url_list = data_url
            content = []
            for url in url_list:
                if to_base64:
                    url = url_to_base64(url)
                content.append({"type": f"image_url", f"image_url": {"url": url}})
            content.append({"type": "text", "text": user_prompt})
        else:
            if to_base64:
                data_url = url_to_base64(data_url)
            content = [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": user_prompt}
            ]
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return content


def openai_server(model_name, data_type, data_url, system_prompt, user_prompt, max_frames, to_base64):
    client = OpenAI(
        api_key=OPENAI_KEY,
        base_url=OPENAI_SERVER_URL,
    )
    dialog_messages = []
    if isinstance(data_url, list):
        print("frames: ", len(data_url))
    content = get_content(model_name, data_type, data_url, user_prompt, max_frames, to_base64=to_base64)
    dialog_messages.append({"role": "system", "content": system_prompt})
    dialog_messages.append({"role": "user", "content": content})
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=dialog_messages,
            stream=True,
        )
        response_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response_content += chunk.choices[0].delta.content
                # print(chunk.choices[0].delta.content, end="")
        # print(response_content)
    except Exception as e:
        print(f'Error: {e} for {data_url} when calling api')
        return None, None

    # try:
    #     response_content = stream.choices[0].message.content
    # except Exception as e:
    #     print(f'Error: {e} for {data_url}, response: {stream.text}')
        
    #     return None, None
    
    return response_content, None

def post_server(model_name, data_type, data_url, system_prompt, user_prompt, max_frames, to_base64):
    time.sleep(random.random() * 5)
    try:
        dialog_messages = []
        content = get_content(model_name, data_type, data_url, user_prompt, max_frames, to_base64=to_base64)
        dialog_messages.append({"role": "system", "content": system_prompt})
        dialog_messages.append({"role": "user", "content": content}) 

        data = {
            "messages": dialog_messages,
            "platformInput": {
                "model": model_name,
                # "isAcquireOriginalOutput": False
            },
        }
        response = requests.post(server_url, headers=headers, data=json.dumps(data))
    except Exception as e:
        print(f'Error: {e} for {data_url} when calling api')
        return None, None
    
    try:
        res_dict = response.json()
    except Exception as e:
        print(f'Error: {e} for {data_url}, response: {response.text}')
        return None, None
    
    try:
        response_content = res_dict.get("data").get("choices")[0].get("message").get("content")
    except Exception as e:
        print(f'Error: {e} for {data_url}, response: {response.text}')
        error_code = res_dict.get("code")
        return None, error_code
    
    return response_content, None


def init_child(lock):
    global global_lock
    global_lock = lock


def handle_CE005(url):
    new_url = url
    retry_times = 0
    while True:
        r = requests.get(new_url)
        file_size = int(r.headers['Content-Length'])
        if file_size <= 3 * 1024 * 1024 or retry_times >= 8:
            break
        print(f"Resize and Retry {retry_times} times")
        retry_times += 1
        resize_ratio = math.sqrt(3 * 1024 * 1024 / file_size)
        assert resize_ratio < 1
        image = Image.open(BytesIO(r.content))
        alpha = 1 if retry_times <= 3 else 0.95
        base = math.floor(max(image.height, image.width) * resize_ratio * alpha)
        resized_image = resize_image(image, base)
        oss_path = 'tmp/{}.jpg'.format(uuid.uuid4().hex)
        quality = 90 if retry_times <= 1 else 80
        new_url = push_resize_image(resized_image, oss_path, quality=quality)
    return new_url


def handle_MPE001(url):
    print("Converting image mode")
    r = requests.get(url)
    image = Image.open(BytesIO(r.content))
    oss_path = 'tmp/{}.{}'.format(uuid.uuid4().hex, url.split('?OSSAccessKeyId')[0].rsplit('.', 1)[1])
    new_url = push_resize_image(image, oss_path, quality=90)
    return new_url


def api_infer_worker(args):
    global global_lock
    server_provider, model_name, data_type, each_data, system_prompt, user_prompt, save_path, error_save_path, max_frames, to_base64 = args
    server_func = openai_server if server_provider == "openai" else post_server
    if "gemini" in model_name:
        url = each_data[f"{data_type}_url"]
    else:
        url = each_data["image_url"] if data_type == "image" else each_data["frames_url"]
    failed_flag = False
    response_content, error_code = server_func(model_name, data_type, url, system_prompt, user_prompt, max_frames, to_base64)
    retry_count = 0
    while response_content is None:
        failed_flag = True
        if error_code is None:
            break
        
        if server_provider != "openai":
            if error_code in ["PL-002", "MPE-429"]:
                response_content, error_code = server_func(model_name, data_type, url, system_prompt, user_prompt, max_frames, to_base64)
            elif data_type == "image" and error_code == "CE-005":  # 资源紧张，单张图片大小不能超过3.0MB
                new_url = handle_CE005(url)
                response_content, error_code = server_func(model_name, data_type, new_url, system_prompt, user_prompt, max_frames, to_base64)
            elif data_type == "image" and error_code == "MPE-001":
                new_url = handle_MPE001(url)
                response_content, error_code = server_func(model_name, data_type, new_url, system_prompt, user_prompt, max_frames, to_base64)
        
            if response_content is not None:
                failed_flag = False
                break
        
        retry_count += 1
        if retry_count > 3:
            break

    if failed_flag:
        error_record = {'file_id': each_data['file_id'], "error_code": error_code}
        if global_lock is None:
            with open(error_save_path, 'a') as f:
                f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
        else:
            with global_lock:
                with open(error_save_path, 'a') as f:
                    f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
        return None

    data = {'file_id': each_data['file_id'], "caption": response_content}

    with nullcontext() if global_lock is None else global_lock:
        with open(save_path, 'a') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    return data


def resize_image(image, base):
    if image.width > image.height:
        # 宽大于高
        w_percent = (base / float(image.width))
        h_size = int((float(image.height) * float(w_percent)))
        if hasattr(Image, 'Resampling'):  # Pillow version >= 9.1.0
            resized_image = image.resize((base, h_size), Image.Resampling.LANCZOS)
        else:  # Pillow version < 9.1.0
            resized_image = image.resize((base, h_size), Image.LANCZOS)
        return resized_image
    else:
        # 高大于宽
        h_percent = (base / float(image.height))
        w_size = int((float(image.width) * float(h_percent)))
        if hasattr(Image, 'Resampling'):  # Pillow version >= 9.1.0
            resized_image = image.resize((w_size, base), Image.Resampling.LANCZOS)
        else:  # Pillow version < 9.1.0
            resized_image = image.resize((w_size, base), Image.LANCZOS)
        return resized_image


def push_resize_image(image, new_object_name, quality):
    output_stream = BytesIO()
    if image.mode != "RGB":
        image = image.convert('RGB')
    image.save(output_stream, format='JPEG', quality=quality)
    output_stream.seek(0)

    # 上传图片到OSS
    bucket.put_object(new_object_name, output_stream)
    url = bucket.sign_url('GET', new_object_name, 60 * 60 * 24 * 1000000)
    return url


def inference(args, dataset_name):
    # read file first and get process_args
    data_root = 'annotations'
    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(args.error_save_root, exist_ok=True)
    video_dataset_name_list = ['dynamic_object_number', 'camera_movement', 'event']
    data_type = 'video' if dataset_name in video_dataset_name_list else 'image'

    ctx = get_context('spawn')
    lock = ctx.Lock()
    
    src_path = f'{data_root}/{dataset_name}.jsonl'
    dest_path = f'{args.save_root}/{dataset_name}.jsonl'
    error_path = f'{args.error_save_root}/{dataset_name}.jsonl'
    with open(error_path, 'w') as f:
        pass
    with open(src_path, 'r') as f:
        src_data = [json.loads(l.strip()) for l in f.readlines()]
    src_file_list = [d['file_id'] for d in src_data]

    if os.path.exists(dest_path):
        with open(dest_path, 'r') as f:
            dest_data = [json.loads(l.strip()) for l in f.readlines()]
        dest_file_list = [d['file_id'] for d in dest_data]
        remain_file_list = [f for f in src_file_list if f not in dest_file_list]
        print(f"[{dataset_name}] Found {len(dest_file_list)} records, remaining {len(remain_file_list)}")
    else:
        remain_file_list = src_file_list
    if len(remain_file_list) == 0:
        os.remove(error_path)
        return True
    remain_data = [d for d in src_data if d['file_id'] in remain_file_list]

    worker_args = []
    system_prompt = model_prompt_dict[args.model_name]["system_prompt"]
    user_prompt = model_prompt_dict[args.model_name][f"{data_type}_prompt"]
    for each_data in remain_data:
        worker_args.append([
            args.server_provider, args.model_name, data_type, each_data,
            system_prompt, user_prompt, dest_path, error_path, args.max_frames, args.to_base64
        ])
    
    if args.num_processes == 0:
        init_child(lock)
        for worker_arg in tqdm(worker_args):
            api_infer_worker(worker_arg)
    else:
        with ctx.Pool(processes=args.num_processes, initializer=init_child, initargs=(lock,)) as pool:
            responses = list(tqdm(pool.imap_unordered(api_infer_worker, worker_args), total=len(worker_args)))
    
    with open(error_path, 'r') as f:
        error_data = [json.loads(l.strip('\n')) for l in f.readlines()]
    if len(error_data) == 0:
        os.remove(error_path)
        return True
    
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=False, default="gemini-1.5-pro")
    parser.add_argument("--save_root", type=str, required=False, default="inference/output/gemini-1.5-pro")
    parser.add_argument("--error_save_root", type=str, required=False, default="inference/error_output/gemini-1.5-pro")
    parser.add_argument("--num_processes", type=int, required=False, default=30)
    parser.add_argument("--max_retry_times", type=int, default=5)
    parser.add_argument("--server_provider", type=str, default="idealab")
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--to_base64", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name_list = [
        'object_category', 'object_number', 'object_color', 'camera_angle',
        'spatial_relation', 'scene', 'style', 'OCR', 'character_identification',
        'dynamic_object_number', 'camera_movement', 'event'
    ]
    for dataset_name in dataset_name_list:
        for i in range(args.max_retry_times):
            success = inference(args, dataset_name)
            if success:
                break



if __name__ == "__main__":
    main()
