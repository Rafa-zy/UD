NUM_GPUS=8

deepspeed --num_gpus=${NUM_GPUS} train.py \
    --model_name_or_path ./huggingface_models/t5-large-lm-adapt \
    --train_file ./train_data/data_cls_5choices_maxlen256.json \
    --output_dir ./results/ud_large \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --max_seq_length 256 \
    --pooler_type avg \
    --overwrite_output_dir \
    --do_train \
    --max_choices 60 \
    --deepspeed ./deepspeed/ds_config_zero2_fp32.json  \
    "$@"