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
from PIL import Image
from utils import CUDADataLoader


def load_video_ffmpeg(
        video_path: str,
        start_time: float = None,
        end_time: float = None,
        fps: float = None,
        max_frames: float = None,
        size: int = None,
        size_divisible: int = 1,
        precise_time: bool = False,
        verbose: bool = False,
        temporal_factor: int = 1
    ):
        """
        Load and process a video file and return the frames and the timestamps of each frame.
        Args:
            video_path (str): Path to the video file.
            start_time (float, optional): Start time in seconds. Defaults to None.
            end_time (float, optional): End time in seconds. Defaults to None.
            fps (float, optional): Frames per second. Defaults to None.
            num_frames (float, optional): Number of frames to sample. Defaults to None.
            size (int, optional): Size of the shortest side. Defaults to None.
            size_divisible (int, optional): Size divisible by this number. Defaults to 1.
            precise_time (bool, optional): Whether to use precise time. Defaults to False.
            verbose (bool, optional): Print ffmpeg output. Defaults to False.
        Returns:
            frames (List[PIL.Image]): List of frames.
            timestamps (List[float]): List of timestamps.
        """
        fps = None
        max_frames = 32

        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        w, h = int(video_stream['width']), int(video_stream['height'])

        kwargs, input_kwargs, output_kwargs = {}, {}, {}
        do_trim = start_time is not None or end_time is not None
        if start_time is not None:
            new_start_time = max(float(video_stream['start_time']), start_time)
            duration -= new_start_time - start_time
            start_time = new_start_time
        else:
            start_time = float(video_stream['start_time'])
        if end_time is not None:
            duration = min(duration, end_time - start_time)
        else:
            duration = duration
        if do_trim:
            kwargs = {'ss': start_time, 't': duration}
        if precise_time:
            output_kwargs.update(kwargs)
        else:
            input_kwargs.update(kwargs)

        if size is not None:
            scale_factor = size / min(w, h)
            new_w, new_h = round(w * scale_factor), round(h * scale_factor)
        else:
            new_w, new_h = w, h
        new_w = new_w // size_divisible * size_divisible
        new_h = new_h // size_divisible * size_divisible

        # NOTE: It may result in unexpected number of frames in ffmpeg
        # if calculate the fps directly according to max_frames
        # if max_frames is not None and (fps is None or duration * fps > 2 * max_frames):
        #     fps = round(max_frames / duration * 2)

        stream = ffmpeg.input(video_path, **input_kwargs)
        if fps is not None:
            stream = ffmpeg.filter(stream, "fps", fps=fps, round="down")
        if new_w != w or new_h != h:
            stream = ffmpeg.filter(stream, 'scale', new_w, new_h)
        stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24", **output_kwargs)
        out, _ = ffmpeg.run(stream, capture_stdout=True, quiet=not verbose)

        frames = np.frombuffer(out, np.uint8).reshape([-1, new_h, new_w, 3]).transpose([0, 3, 1, 2])

        if max_frames is not None and len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = frames[indices]

        if temporal_factor > 1:
            pad_length = temporal_factor - len(frames) % temporal_factor
            frames = np.concatenate([frames, frames[-1:].repeat(pad_length, axis=0)])

        frames = frames.transpose(0,2,3,1)

        return frames


def load_video(video_path, max_frames_num=32, fps=1, force_sample=True):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=2)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


class EvalDataset(Dataset):
    def __init__(self, data_list, processor, num_splits, split_idx, complex_prompt=False):
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
        self.processor = processor

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        case = self.data_list[index]

        if case['data_type'] == 'image':
            file_path = "../" + case["file_path"]
            query = self.image_prompt
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query},
                    ]
                }
            ]
            image = Image.open(file_path)
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(torch.float16)
        else:
            file_path = "../" + case["file_path"]
            query = self.video_prompt
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": query},
                    ]
                }
            ]
            video = load_video(file_path)
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(videos=video, text=prompt, return_tensors='pt').to(torch.float16)

        return case, inputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=False, default="event")
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--save_root", type=str, required=False)
    parser.add_argument("--complex_prompt", action='store_true', default=False)

    return parser.parse_args()


def main():
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()

    args = parse_args()
    model_name = args.model_path

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map={"": f"cuda:{local_rank}"},
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

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

    dataset = EvalDataset(meta, processor=processor, num_splits=dist.get_world_size(), split_idx=global_rank, complex_prompt=args.complex_prompt)
    dataloader = CUDADataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        collate_fn=lambda x: x[0],  # asume the batch_size is always 1
        pin_memory=True,
    )

    generate_kwargs = {"max_new_tokens": 8196, "do_sample": False}
    for n, (case, inputs) in enumerate(tqdm(dataloader, desc=f"Rank {global_rank}", total=len(dataloader), position=local_rank)):
        output = model.generate(**inputs, **generate_kwargs)
        response = processor.batch_decode(output, skip_special_tokens=True)[0].strip().split("assistant\n")[-1]

        with open(f"{args.save_root}/{ann}", 'a', encoding='utf-8') as f:
            f.write(json.dumps({'file_id': case['file_id'], 'caption': response}, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
