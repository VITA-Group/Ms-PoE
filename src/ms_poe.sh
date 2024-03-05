for answer_idx in $1; do
    GPU=0
    SEED=42
    NUM_DOCUMENTS=10
    models=lmsys/vicuna-7b-v1.5
    name=vicuna_7b    
    min_ratio=1.2
    max_ratio=1.8
    output_path=mdqa_results/ours-${name}-${NUM_DOCUMENTS}doc-answer${answer_idx}-ratio${min_ratio}to${max_ratio}-Seed${SEED}-2to31.jsonl

    CUDA_VISIBLE_DEVICES=${GPU} python -u inference.py \
        --input_path data/mdqa_${NUM_DOCUMENTS}documents.jsonl.gz \
        --output_path ${output_path} \
        --model_name ${models} \
        --seed ${SEED} \
        --sample_num 500 \
        --answer_idx ${answer_idx} \
        --enable_ms_poe \
        --apply_layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
        --compress_ratio_min ${min_ratio} \
        --compress_ratio_max ${max_ratio} \
        --cache_dir ../llm_weights # server-dependent setting

    python -u utils/lost_in_the_middle/eval_qa_response.py --input-path ${output_path}
done







