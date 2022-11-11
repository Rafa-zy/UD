write_dir="../our_result/results725_t5_large_cls_old_5choices_maxlen256"

mkdir -p ${write_dir}

python eval.py \
    --model_name_or_path result/our_t5_large_cls_old_5choices_725/checkpoint-3500 \
    --write_path ${write_dir}/epoch1.csv \
    --pooler_type avg \
    --temp 0.05 \
    "$@"
