import os
import json

folder = "llava_ov_72b_hf"

dataset_list = [
    'object_category', 'object_number', 'dynamic_object_number', 'object_color',
    'spatial_relation', 'scene', 'style', 'OCR', 'character_identification',
    'camera_angle', 'camera_movement',  'event'
]

anno_root = "annotations"

response_root = f"inference/output/{folder}"

for dataset in dataset_list:
    with open(os.path.join(anno_root, f"{dataset}.jsonl")) as f:
        anno = [json.loads(line.strip()) for line in f.readlines()]
    anno_ids = [a["file_id"] for a in anno]
    assert len(set(anno_ids)) == len(anno_ids), f"Duplicate gt file_id in {dataset}.jsonl"

    with open(os.path.join(response_root, f"{dataset}.jsonl")) as f:
        response = [json.loads(line.strip()) for line in f.readlines()]
    response_ids = [r["file_id"] for r in response]
    assert len(set(response_ids)) == len(response_ids), f"Duplicate response file_id in {dataset}.jsonl"

    for rid in response_ids:
        assert rid in anno_ids, f"Unknown file_id {rid} in Response {dataset}.jsonl"
    if len(anno_ids) > len(response_ids):
        print(f"!!!!!!!!!!![{dataset}] All {len(anno_ids)}, Remaining {len(anno_ids) - len(response_ids)}")
    else:
        print(f"[{dataset}] All {len(anno_ids)}, Remaining {len(anno_ids) - len(response_ids)}")


    