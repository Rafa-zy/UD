: '
python discriminator_demo.py \
    --model_name_or_path ./results/ud_large \
    --pooler_type avg \
    "$@"
'

python discriminator_demo.py \
    --model_name_or_path ./UD_ckpt/our_t5_large_data520_cls_old_5choices_maxlen256_edited_maxseqlen256/checkpoint-22000 \
    --pooler_type avg \
    "$@"