CUDA_VISIBLE_DEVICES=0 python -u inference.py \
    --input_path data/mdqa_10documents.jsonl.gz \
    --output_path mdqa_results/baseline-vicuna_7b-10doc-answer1.jsonl \
    --model_name lmsys/vicuna-7b-v1.5 \
    --seed 42 \
    --sample_num 500 \
    --answer_idx 1
python -u utils/lost_in_the_middle/eval_qa_response.py --input-path mdqa_results/baseline-vicuna_7b-10doc-answer1.jsonl


CUDA_VISIBLE_DEVICES=0 python -u inference.py \
    --input_path data/mdqa_10documents.jsonl.gz \
    --output_path mdqa_results/baseline-vicuna_7b-10doc-answer3.jsonl \
    --model_name lmsys/vicuna-7b-v1.5 \
    --seed 42 \
    --sample_num 500 \
    --answer_idx 3
python -u utils/lost_in_the_middle/eval_qa_response.py --input-path mdqa_results/baseline-vicuna_7b-10doc-answer3.jsonl


CUDA_VISIBLE_DEVICES=0 python -u inference.py \
    --input_path data/mdqa_10documents.jsonl.gz \
    --output_path mdqa_results/baseline-vicuna_7b-10doc-answer5.jsonl \
    --model_name lmsys/vicuna-7b-v1.5 \
    --seed 42 \
    --sample_num 500 \
    --answer_idx 5
python -u utils/lost_in_the_middle/eval_qa_response.py --input-path mdqa_results/baseline-vicuna_7b-10doc-answer5.jsonl


CUDA_VISIBLE_DEVICES=0 python -u inference.py \
    --input_path data/mdqa_10documents.jsonl.gz \
    --output_path mdqa_results/baseline-vicuna_7b-10doc-answer7.jsonl \
    --model_name lmsys/vicuna-7b-v1.5 \
    --seed 42 \
    --sample_num 500 \
    --answer_idx 7
python -u utils/lost_in_the_middle/eval_qa_response.py --input-path mdqa_results/baseline-vicuna_7b-10doc-answer7.jsonl


CUDA_VISIBLE_DEVICES=0 python -u inference.py \
    --input_path data/mdqa_10documents.jsonl.gz \
    --output_path mdqa_results/baseline-vicuna_7b-10doc-answer10.jsonl \
    --model_name lmsys/vicuna-7b-v1.5 \
    --seed 42 \
    --sample_num 500 \
    --answer_idx 10
python -u utils/lost_in_the_middle/eval_qa_response.py --input-path mdqa_results/baseline-vicuna_7b-10doc-answer10.jsonl


