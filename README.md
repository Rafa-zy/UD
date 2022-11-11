# README

First, produce the data for our training process:

```python
python produce_data.py
```

Then, train our main model (i.e., Universal Discriminator).

You can directly use the following script:

```python
bash train.sh
```

Or you can run by changing some of the parameters for usage:

```python
NUM_GPUS=8

deepspeed --num_gpus=${NUM_GPUS} train.py \
    --model_name_or_path ./huggingface_models/t5-large-lm-adapt \
    --train_file ../data_for_simcse/data725_cls_old_5choices_maxlen256_unidir.json \
    --output_dir ./result/our_t5_large_cls_old_5choices_725 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --max_seq_length 256 \
    --pooler_type avg \
    --overwrite_output_dir \
    --do_train \
    --max_choices 60 \
    --deepspeed ../deepspeed/ds_config_zero2_fp32.json  \
    "$@"
```

Finally, eval T0 tasks using our trained checkpoint:

Also, you can directly use the following script:

```python
bash eval.sh
```

Or you can run by changing some of the parameters for usage:

```python
write_dir="../our_result/results725_t5_large_cls_old_5choices_maxlen256"

mkdir -p ${write_dir}

python eval.py \
    --model_name_or_path result/our_t5_large_cls_old_5choices_725/checkpoint-3500 \
    --write_path ${write_dir}/epoch1.csv \
    --pooler_type avg \
    --temp 0.05 \
    "$@"
```

Noted that you should prepare the T5 checkpoints and T0's training and evaluation task data mentioned in paper to run through the above code.