set -x

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

# image
benchmarks=(
    "object_category"
    "object_number"
    # "dynamic_object_number"
    "object_color"
    "spatial_relation"
    "scene"
    "style"
    "OCR"
    "character_identification"
    "camera_angle"
    # "camera_movement"
    # "event"
)

for BENCHMARK in "${benchmarks[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master_addr=127.0.0.1 --master_port=16668 --node_rank 0 \
        internvl_8b.py --benchmark ${BENCHMARK} \
        --model_path models/InternVL2_5-8B \
        --save_root output/internvl2.5_8b \
        --complex_prompt

    CUDA_VISIBLE_DEVICES=0,1,2,3 python internvl_78b.py \
        --benchmark ${BENCHMARK} \
        --model_path models/InternVL2_5-78B \
        --save_root output/internvl2.5_78b \
        --complex_prompt

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master_addr=127.0.0.1 --master_port=16668 --node_rank 0 \
        llava_ov_hf_7b.py --benchmark ${BENCHMARK} \
        --model_path models/llava-onevision-qwen2-7b-ov-hf \
        --save_root output/llava_ov_7b_hf \
        --complex_prompt

    CUDA_VISIBLE_DEVICES=0,1,2,3 python llava_ov_hf_72b.py \
        --benchmark ${BENCHMARK} \
        --model_path models/llava-onevision-qwen2-72b-ov-hf \
        --save_root output/llava_ov_72b_hf \
        --complex_prompt

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master_addr=127.0.0.1 --master_port=16668 --node_rank 0 \
        nvila_8b.py --benchmark ${BENCHMARK} \
        --model_path models/NVILA-8B \
        --save_root output/nvila_8b \
        --complex_prompt
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master_addr=127.0.0.1 --master_port=16668 --node_rank 0 \
        videollama3.py --benchmark ${BENCHMARK} \
        --model_path models/VideoLLaMA3-7B-Image \
        --save_root output/videollama3_7b \
        --complex_prompt
done


# video
benchmarks=(
    # "object_category"
    # "object_number"
    "dynamic_object_number"
    # "object_color"
    # "spatial_relation"
    # "scene"
    # "style"
    # "OCR"
    # "character_identification"
    # "camera_angle"
    "camera_movement"
    "event"
)

for BENCHMARK in "${benchmarks[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master_addr=127.0.0.1 --master_port=16668 --node_rank 0 \
        internvl_8b.py --benchmark ${BENCHMARK} \
        --model_path models/InternVL2_5-8B \
        --save_root output/internvl2.5_8b \
        --complex_prompt

    CUDA_VISIBLE_DEVICES=0,1,2,3 python internvl_78b.py \
        --benchmark ${BENCHMARK} \
        --model_path models/InternVL2_5-78B \
        --save_root output/internvl2.5_78b \
        --complex_prompt

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master_addr=127.0.0.1 --master_port=16668 --node_rank 0 \
        llava_ov_hf_7b.py --benchmark ${BENCHMARK} \
        --model_path models/llava-onevision-qwen2-7b-ov-hf \
        --save_root output/llava_ov_7b_hf \
        --complex_prompt

    CUDA_VISIBLE_DEVICES=0,1,2,3 python llava_ov_hf_72b.py \
        --benchmark ${BENCHMARK} \
        --model_path models/llava-onevision-qwen2-72b-ov-hf \
        --save_root output/llava_ov_72b_hf \
        --complex_prompt

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master_addr=127.0.0.1 --master_port=16668 --node_rank 0 \
        nvila_8b.py --benchmark ${BENCHMARK} \
        --model_path models/NVILA-8B \
        --save_root output/nvila_8b \
        --complex_prompt
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master_addr=127.0.0.1 --master_port=16668 --node_rank 0 \
        videollama3.py --benchmark ${BENCHMARK} \
        --model_path models/VideoLLaMA3-7B \
        --save_root output/videollama3_7b \
        --complex_prompt
done


CUDA_VISIBLE_DEVICES=0,1,2,3 python qwenvl_vllm.py --num_gpus 4 \
    --model_path models/Qwen2.5-VL-7B-Instruct \
    --save_root output/qwen2.5vl_7b \
    --error_save_root output/error_cases/qwen2.5vl_7b \
    --complex_prompt 

CUDA_VISIBLE_DEVICES=0,1,2,3 python qwenvl_vllm.py --num_gpus 4 \
    --model_path models/Qwen2.5-VL-72B-Instruct \
    --save_root output/qwen2.5vl_72b \
    --error_save_root output/error_cases/qwen2.5vl_72b \
    --complex_prompt 

CUDA_VISIBLE_DEVICES=0,1,2,3 python qwenvl_vllm.py --num_gpus 4 \
    --model_path models/Qwen2-VL-7B-Instruct \
    --save_root output/qwen2vl_7b \
    --error_save_root output/error_cases/qwen2vl_7b \
    --complex_prompt

CUDA_VISIBLE_DEVICES=0,1,2,3 python qwenvl_vllm.py --num_gpus 4 \
    --model_path models/Qwen2-VL-72B-Instruct \
    --save_root output/qwen2vl_72b \
    --error_save_root output/error_cases/qwen2vl_72b \
    --complex_prompt
