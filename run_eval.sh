write_dir="./evaluation_results/ud_large"

mkdir -p ${write_dir}

python eval.py \
    --model_name_or_path ./results/ud_large \
    --write_path ${write_dir}/final.csv \
    --pooler_type avg \
    "$@"
