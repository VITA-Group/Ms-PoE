for answer_idx in $1; do
    GPU=0
    SEED=42
    NUM_DOCUMENTS=10
    models=lmsys/vicuna-7b-v1.5
    name=vicuna_7b
    output_path=mdqa_results/baseline-${name}-${NUM_DOCUMENTS}doc-answer${answer_idx}-Seed${SEED}.jsonl

    CUDA_VISIBLE_DEVICES=${GPU} python -u inference.py \
        --input_path data/mdqa_${NUM_DOCUMENTS}documents.jsonl.gz \
        --output_path ${output_path} \
        --model_name ${models} \
        --seed ${SEED} \
        --sample_num 500 \
        --answer_idx ${answer_idx} \
        --cache_dir ../llm_weights # server-dependent setting

    python -u utils/lost_in_the_middle/eval_qa_response.py --input-path ${output_path}
done


