write_dir="./evaluation_results/ud_large"

mkdir -p ${write_dir}
# you should move the ckpt to the following path
python eval.py \
    --model_name_or_path ./results/ud_large/checkpoint-22000 \
    --write_path ${write_dir}/final.csv \
    --pooler_type avg \
    "$@"
