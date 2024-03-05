
CUDA_VISIBLE_DEVICES=0 python -u inference.py \
    --input_path data/mdqa_10documents.jsonl.gz \
    --output_path mdqa_results/ours-vicuna_7b-10doc-answer1-ratio1.2to1.8.jsonl \
    --model_name lmsys/vicuna-7b-v1.5 \
    --seed 42\
    --sample_num 500 \
    --answer_idx 1 \
    --enable_ms_poe \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
    --compress_ratio_min 1.2 \
    --compress_ratio_max 1.8 
python -u utils/lost_in_the_middle/eval_qa_response.py --input-path mdqa_results/ours-vicuna_7b-10doc-answer1-ratio1.2to1.8.jsonl


CUDA_VISIBLE_DEVICES=0 python -u inference.py \
    --input_path data/mdqa_10documents.jsonl.gz \
    --output_path mdqa_results/ours-vicuna_7b-10doc-answer3-ratio1.2to1.8.jsonl \
    --model_name lmsys/vicuna-7b-v1.5 \
    --seed 42\
    --sample_num 500 \
    --answer_idx 3 \
    --enable_ms_poe \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
    --compress_ratio_min 1.2 \
    --compress_ratio_max 1.8 
python -u utils/lost_in_the_middle/eval_qa_response.py --input-path mdqa_results/ours-vicuna_7b-10doc-answer3-ratio1.2to1.8.jsonl



CUDA_VISIBLE_DEVICES=0 python -u inference.py \
    --input_path data/mdqa_10documents.jsonl.gz \
    --output_path mdqa_results/ours-vicuna_7b-10doc-answer5-ratio1.2to1.8.jsonl \
    --model_name lmsys/vicuna-7b-v1.5 \
    --seed 42\
    --sample_num 500 \
    --answer_idx 5 \
    --enable_ms_poe \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
    --head_type "normal" \
    --compress_ratio_min 1.2 \
    --compress_ratio_max 1.8 
python -u utils/lost_in_the_middle/eval_qa_response.py --input-path mdqa_results/ours-vicuna_7b-10doc-answer5-ratio1.2to1.8.jsonl


CUDA_VISIBLE_DEVICES=0 python -u inference.py \
    --input_path data/mdqa_10documents.jsonl.gz \
    --output_path mdqa_results/ours-vicuna_7b-10doc-answer7-ratio1.2to1.8.jsonl \
    --model_name lmsys/vicuna-7b-v1.5 \
    --seed 42\
    --sample_num 500 \
    --answer_idx 7 \
    --enable_ms_poe \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
    --head_type "normal" \
    --compress_ratio_min 1.2 \
    --compress_ratio_max 1.8 
python -u utils/lost_in_the_middle/eval_qa_response.py --input-path mdqa_results/ours-vicuna_7b-10doc-answer7-ratio1.2to1.8.jsonl


CUDA_VISIBLE_DEVICES=0 python -u inference.py \
    --input_path data/mdqa_10documents.jsonl.gz \
    --output_path mdqa_results/ours-vicuna_7b-10doc-answer10-ratio1.2to1.8.jsonl \
    --model_name lmsys/vicuna-7b-v1.5 \
    --seed 42\
    --sample_num 500 \
    --answer_idx 10 \
    --enable_ms_poe \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
    --head_type "normal" \
    --compress_ratio_min 1.2 \
    --compress_ratio_max 1.8 
python -u utils/lost_in_the_middle/eval_qa_response.py --input-path mdqa_results/ours-vicuna_7b-10doc-answer10-ratio1.2to1.8.jsonl





