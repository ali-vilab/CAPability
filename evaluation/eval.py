import os
import ast
import time
import json
import shutil
import random
import requests
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from functools import partial
from contextlib import nullcontext
from multiprocessing import Pool, Lock, get_context

from prompt import Prompts


SERVER_URL = os.environ["SERVER_URL"]
API_KEY = os.environ["API_KEY"]
# HEADERS = {
#     'Authorization': f'Bearer {API_KEY}',
#     'Content-Type': 'application/json'
# }


LOCAL_API_URL = "http://localhost:8000/v1"

global_lock = None


class Evaluator:
    def __init__(
            self, caption_file_root, gt_file_root, save_root,
            tasks="all", num_process=0,
            eval_model="gpt-4-turbo", max_allow_missing=5, 
            auto_resume=True, max_retry_times=10, strict_match=True,
    ):
        if isinstance(tasks, list):
            self.tasks = tasks
        elif tasks == "all":
            self.tasks = [
                "object_category", "object_number", "object_color", "spatial_relation", 
                "scene", "camera_angle", "OCR", "style", "character_identification", 
                "dynamic_object_number", "action", "camera_movement", "event"
            ]
        elif isinstance(tasks, str):
            self.tasks = tasks.split(',')
        
        self.captions = {}
        self.annotations = {}
        task2cap_filename = {r.split('samples_capability_')[-1].split('.jsonl')[0]: r for r in os.listdir(caption_file_root) if r.endswith('jsonl')}
        for task in self.tasks:
            
            if task == "action" and task not in task2cap_filename:
                shutil.copyfile(
                    os.path.join(caption_file_root, task2cap_filename['event']),
                    os.path.join(caption_file_root, task2cap_filename['event'].replace('event', 'action'))
                )
                task2cap_filename['action'] = task2cap_filename['event'].replace('event', 'action')
            cap_filename = task2cap_filename[task]
            with open(os.path.join(caption_file_root, cap_filename), 'r') as f:
                self.captions[task] = [json.loads(l.strip('\n')) for l in f.readlines()]
            with open(os.path.join(gt_file_root, f"{task}.jsonl"), 'r') as f:
                self.annotations[task] = [json.loads(l.strip('\n')) for l in f.readlines()]
            
            del_keys = [k for k in self.captions[task][0].keys() if k not in ["file_id", "caption"]]
            for i in range(len(self.captions[task])):
                for k in del_keys:
                    del self.captions[task][i][k]
        
        self.save_root = save_root
        os.makedirs(self.save_root, exist_ok=True)
        self.num_process = num_process
        self.eval_model = eval_model
        self.max_allow_missing = max_allow_missing
        self.auto_resume = auto_resume
        self.max_retry_times = max_retry_times
        self.strict_match = strict_match
        self.prompts = Prompts()

        self.post_validate_format_func = {}
        self.post_process_func = {}
        for task in self.tasks:
            self.post_validate_format_func[task] = eval(f"self.post_validate_format_{task}")
            self.post_process_func[task] = eval(f"self.post_process_{task}")
    
    @staticmethod
    def call_gpt(eval_model, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            client = OpenAI(
                api_key=API_KEY,
                base_url=SERVER_URL,
            )
            response = client.chat.completions.create(
                model=eval_model,
                messages=messages,
            )
            response_message = response.choices[0].message.content
            return response_message

        except Exception as e:
            print(f"\tError calling {eval_model}: {e}") # \n\tResponse: {response}")
            return None
        
    def load_saved_records(self, save_path):
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                saved_responses = [json.loads(l.strip('\n')) for l in f.readlines()]
            print(f"Loaded {len(saved_responses)} records")
        else:
            saved_responses = []
        return saved_responses

    def post_validate_format_event(self, response, anno):
        # "{\"action\": \"copy provided action here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if self.strict_match:
            assert response["event"].strip() == anno.strip()
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_event(self, response, anno):
        return response["score"]
    
    def post_validate_format_action(self, response, anno):
        # "{\"action\": \"copy provided action here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if self.strict_match:
            assert response["action"].strip() == anno.strip()
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_action(self, response, anno):
        return response["score"]

    def post_validate_format_object_category(self, response, anno):
        # "{\"object_category\": \"copy provided object here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if self.strict_match:
            assert response["object_category"].strip() == anno.strip()
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_object_category(self, response, anno):
        return response["score"]
    
    def post_validate_format_object_number(self, response, anno):
        # "{\"object_number\": \"copy the provided {object: number} here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if isinstance(response['object_number'], str):
            # assert response['object_number'].startswith("{") and response['object_number'].endswith("}")
            assert ':' in response['object_number']
            object_category, object_number = response['object_number'].lstrip('{').rstrip('}').split(":")
            object_number = int(object_number.strip())
        elif isinstance(response['object_number'], dict):
            object_category, object_number = list(response['object_number'].items())[0]
            object_number = int(object_number.strip())
        else:
            raise ValueError("Invalid object_number format")
        if self.strict_match:
            assert object_number == list(anno.values())[0]
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_object_number(self, response, anno):
        return response["score"]

    def post_validate_format_dynamic_object_number(self, response, anno):
        # "{\"object_number\": \"copy the provided {object: number} here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        assert 'response' in response
        for i, r in enumerate(response['response']):
            if isinstance(r['object_number'], str):
                # assert response['object_number'].startswith("{") and response['object_number'].endswith("}")
                assert ':' in r['object_number']
                object_category, object_number = r['object_number'].lstrip('{').rstrip('}').split(":")
                object_number = int(object_number.strip())
            elif isinstance(response['object_number'], dict):
                object_category, object_number = list(r['object_number'].items())[0]
                object_number = int(object_number.strip())
            else:
                raise ValueError("Invalid object_number format")
            if self.strict_match:
                assert object_number == list(anno.values())[i]
            if r["score"] in ["-1", "0", "1"]:
                r["score"] = int(r["score"])
            assert r["score"] in [1, 0, -1]

    def post_process_dynamic_object_number(self, response, anno):
        scores = []
        for r in response['response']:
            scores.append(r['score'])
        return scores

    def post_validate_format_object_color(self, response, anno):
        # "{\"object_color\": \"copy the provided {object: color} here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if isinstance(response['object_color'], str):
            # assert response['object_color'].startswith("{") and response['object_color'].endswith("}")
            assert ':' in response['object_color']
            unpacked = response['object_color'].lstrip('{').rstrip('}').split(":")
            if len(unpacked) > 2:
                object_category, object_color = ":".join(unpacked[:-1]), unpacked[-1]
            else:
                object_category, object_color = unpacked
            object_color = object_color.strip()
        elif isinstance(response['object_color'], dict):
            object_category, object_color = list(response['object_color'].items())[0]
            object_color = object_color.strip()
        else:
            raise ValueError("Invalid object_color format")
        if self.strict_match:
            assert object_color == list(anno.values())[0]
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_object_color(self, response, anno):
        return response["score"]

    def post_validate_format_spatial_relation(self, response, anno):
        # "{\"spatial_relation\": \"copy the provided spatial relationship here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if self.strict_match:
            assert response["spatial_relation"].strip() == anno.strip()
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_spatial_relation(self, response, anno):
        return response["score"]

    def post_validate_format_scene(self, response, anno):
        # "{\"scene\": \"copy the provided scene here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if self.strict_match:
            assert response["scene"].strip() == anno.strip()
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_scene(self, response, anno):
        return response["score"]

    def post_validate_format_camera_angle(self, response, anno):
        # "{\"pred\": \"put your predicted category here\", \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        assert "pred" in response
        if response["pred"] == "N/A" or "N/A" in response["pred"]:
            response["pred"] = ["N/A"]
        if isinstance(response["pred"], str):
            response["pred"] = ast.literal_eval(response['pred'])
        assert isinstance(response["pred"], list)
        for i in range(len(response["pred"])):
            if response["pred"][i] in self.prompts.camera_angle_category_explains:
                response["pred"][i] = response["pred"].split(":")[0].lower()
            assert response["pred"][i] == "N/A" or response["pred"][i] in self.prompts.camera_angle_categories
    
    def post_process_camera_angle(self, response, anno):
        if len(response["pred"]) == 1 and response["pred"][0] == "N/A":
            return 0
        elif anno in response["pred"]:
            return 1
        else:
            return -1

    def post_validate_format_camera_movement(self, response, anno):
        # "{\"pred\": \"put your predicted category here\", \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        assert "pred" in response
        if response["pred"] == "N/A" or "N/A" in response["pred"]:
            response["pred"] = ["N/A"]
        if isinstance(response["pred"], str):
            response["pred"] = ast.literal_eval(response['pred'])
        assert isinstance(response["pred"], list)
        for i in range(len(response["pred"])):
            if response["pred"][i] in self.prompts.camera_movement_category_explains:
                response["pred"][i] = response["pred"].split(":")[0].lower()
            assert response["pred"][i] == "N/A" or response["pred"][i] in self.prompts.camera_movement_categories
    
    def post_process_camera_movement(self, response, anno):
        if len(response["pred"]) == 1 and response["pred"][0] == "N/A":
            return 0
        elif anno in response["pred"]:
            return 1
        else:
            return -1

    def post_validate_format_OCR(self, response, anno):
        # "{\"OCR\": \"copy the provided real OCR text here\", \"score\": put your score here, \"reason\": \"give your reason here\"},\n"\
        assert isinstance(response, dict)
        if self.strict_match:
            assert response['OCR'].strip() == anno.strip()
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]
        
    def post_process_OCR(self, response, anno):
        return response['score']

    def post_validate_format_style(self, response, anno):
        # "{\"pred\": \"put your predicted category here\", \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        assert "pred" in response
        if response["pred"] == "N/A" or "N/A" in response["pred"]:
            response["pred"] = ["N/A"]
        if isinstance(response["pred"], str):
            response["pred"] = ast.literal_eval(response['pred'])
        assert isinstance(response["pred"], list)
        for i in range(len(response["pred"])):
            if response["pred"][i] in self.prompts.style_category_explains:
                response["pred"][i] = response["pred"][i].split(":")[0].lower()
            assert response["pred"][i] == "N/A" or response["pred"][i] in self.prompts.style_categories

    def post_process_style(self, response, anno):
        if len(response["pred"]) == 1 and response["pred"][0] == "N/A":
            return 0
        elif anno in response["pred"]:
            return 1
        else:
            return -1
    
    def post_validate_format_character_identification(self, response, anno):
        # "{\"name\": \"copy the provided name here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if self.strict_match:
            assert response["character_identification"].strip() == anno.strip()
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_character_identification(self, response, anno):
        return response["score"]
    
    def intialize_data_by_task(self, task):
        captions = self.captions[task]
        annotations = self.annotations[task]
        self.data_type = annotations[0]['data_type']
        
        self.file2caption = {d[f"file_id"]: d["caption"] for d in captions}
        self.file2anno = {d[f"file_id"]: d["annotation"] for d in annotations}
        if len(self.file2anno) < len(self.file2caption):
            assert task == "action"
            self.file2caption = {k: v for k, v in self.file2caption.items() if k in self.file2anno}

        assert set(self.file2caption.keys()) == set(self.file2anno.keys())
        self.file_list = list(self.file2anno.keys())

    @staticmethod
    def init_child(lock):
        global global_lock
        global_lock = lock

    @staticmethod
    def call_and_parse_single_meaasge(file, system_prompt, user_prompt, eval_model):
        response_message = Evaluator.call_gpt(eval_model, system_prompt, user_prompt)
        if response_message is None:
            print(f"\tCalling GPT failed for {file}")
            return None

        try:
            if '```json' in response_message:
                response_message = response_message.split('```json')[-1].split('```')[0].strip()
            if '```python' in response_message:
                response_message = response_message.split('```python')[-1].split('```')[0].strip()
            elif '```' in response_message:
                response_message = response_message.split('```')[1].strip()
            response = ast.literal_eval(response_message)
            return response
        except (SyntaxError, ValueError) as e:
            print(f"\tInvalid response format for {file}: {response_message}")
            return None

    @staticmethod
    def evaluate_sample_worker(args):
        file, anno, system_prompt, user_prompt, eval_model, post_validate, save_path = args
        global global_lock
        if isinstance(user_prompt, list):
            response = {'response': []}
            for prompt in user_prompt:
                single_response = Evaluator.call_and_parse_single_meaasge(file, system_prompt, prompt, eval_model)
                if single_response is None:
                    return None
                response['response'].append(single_response)
            
        else:
            response = Evaluator.call_and_parse_single_meaasge(file, system_prompt, user_prompt, eval_model)
            if response is None:
                return None
        
        try:
            post_validate(response, anno)
        except Exception as e:
            print(f"\tFormat validation failed for {file}: {e}, anno: {anno}, response: {response}")
            return None

        response['file_id'] = file

        with nullcontext() if global_lock is None else global_lock:
            with open(save_path, 'a') as f:
                f.write(json.dumps(response) + '\n')

        return response

    def evaluate_scores_by_task(self, task):
        self.intialize_data_by_task(task)
        score_dict = {}
        # Load saved records for resuming evaluation
        save_path = os.path.join(self.save_root, f"{task}.jsonl")
        if self.auto_resume:
            saved_responses = self.load_saved_records(save_path)
            print(f"[{task}] Loaded {len(saved_responses)} records")
        else:
            saved_responses = []
        
        # Evaluate remaining
        for retry_count in range(self.max_retry_times + 1):
            saved_files = [r['file_id'] for r in saved_responses]
            if len(saved_files) == len(self.file_list):
                break
            if len(self.file_list) - len(saved_files) <= self.max_allow_missing:
                break

            remaining_files = [v for v in self.file_list if v not in saved_files]
            if retry_count != 0:
                print(f"\nRetrying {retry_count} times")
            
            process_args = []
            for file in remaining_files:
                caption = self.file2caption[file]
                anno = self.file2anno[file]
                system_prompt, user_prompt = self.prompts.get_prompts_by_task(task, caption, anno)
                args = (
                    file, anno, system_prompt, user_prompt, self.eval_model,
                    self.post_validate_format_func[task], save_path
                )
                process_args.append(args)
            
            if self.num_process == 0:
                responses = []
                for args in tqdm(process_args, desc=f"Evaluating {task}"):
                    responses.append(self.evaluate_sample_worker(args))
                    # if response is not None:
                    #     saved_responses.append(response)
                    #     score_dict[response['file_id']] = self.post_process_func[task](response, anno)
            else:
                ctx = get_context('spawn')
                lock = ctx.Lock()
                with ctx.Pool(processes=self.num_process, initializer=self.init_child, initargs=(lock,)) as pool:
                    responses = list(tqdm(pool.imap_unordered(self.evaluate_sample_worker, process_args), total=len(remaining_files), desc=f"Evaluating {task}"))
            responses = [r for r in responses if r is not None]
            saved_responses += responses
        
        for response in tqdm(saved_responses, desc=f"Calculating {task} scores"):
            file = response['file_id']
            score_dict[file] = self.post_process_func[task](response, self.file2anno[file])
            
        return score_dict

    def calculate_metric(self, task, score_dict):
        all_scores = []
        for vid, scores in score_dict.items():
            if isinstance(scores, list):
                all_scores += scores
            else:
                all_scores.append(scores)
        all_scores = np.array(all_scores)
        sum_count = len(all_scores)
        hit_count = np.count_nonzero(all_scores != 0)
        correct_count = np.count_nonzero(all_scores == 1)
        precision = 100 * correct_count / hit_count
        recall = 100 * correct_count / sum_count
        hit_rate = 100 * hit_count / sum_count
        f1_score = 2 * precision * recall / (precision + recall)
        print(f"[{task}] all: {sum_count}, hit: {hit_count}, correct: {correct_count}; precision: {precision:.1f}, recall: {recall:.1f}, f1_score: {f1_score:.1f}, hit_rate: {hit_rate:.1f}\n")
        return {
            "precision": precision,
            "recall": recall,
            "hit_rate": hit_rate,
            "f1_score": f1_score
        }
    
    def evaluate(self):
        all_score_dict = {}
        for t in self.tasks:
            all_score_dict[t] = self.evaluate_scores_by_task(t)
            metric = self.calculate_metric(t, all_score_dict[t])
        
        # summarize metrics
        print("Summarized Results:")
        metrics = []
        for t in self.tasks:
            metric = self.calculate_metric(t, all_score_dict[t])
            metrics.append(metric)
        avg_precision = np.mean([m["precision"] for m in metrics])
        avg_recall = np.mean([m["recall"] for m in metrics])
        avg_hit_rate = np.mean([m["hit_rate"] for m in metrics])
        avg_f1_score = np.mean([m["f1_score"] for m in metrics])
        print(f"Average precision: {avg_precision:.3f}, recall: {avg_recall:.3f}, f1_score: {avg_f1_score:.3f}, hit_rate: {avg_hit_rate:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate video captioning.")
    parser.add_argument("--caption_file_root", default='inference/output/gemini-1.5-pro')
    parser.add_argument("--gt_file_root", default='annotations')
    parser.add_argument("--save_root", default='evaluation/output/gemini-1.5-pro')
    parser.add_argument("--tasks", default='all')
    parser.add_argument("--num_process", type=int, default=0)
    parser.add_argument("--eval_model", default="gpt-4-turbo")
    parser.add_argument("--max_retry_times", type=int, default=5)
    parser.add_argument("--max_allow_missing", type=int, default=5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluator = Evaluator(
        caption_file_root=args.caption_file_root,
        gt_file_root=args.gt_file_root,
        save_root=args.save_root,
        tasks=args.tasks,
        num_process=args.num_process,
        eval_model=args.eval_model,
        max_allow_missing=args.max_allow_missing,
        auto_resume=True,
        max_retry_times=args.max_retry_times,
        strict_match=False,
    )
    evaluator.evaluate()