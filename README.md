# README of UD

First, download and unzip our training data (T0 training data sets with minimal prompts) and evaluation data (T0 evaluating data sets) and put them in folders "train_data" and "eval_data" respectively.

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
    --train_file ../data_for_simcse/data_cls_5choices_maxlen256.json \
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
write_dir="./evaluation_results/ud_large"

mkdir -p ${write_dir}

python eval.py \
    --model_name_or_path results/ud_large/ \
    --write_path ${write_dir}/final.csv \
    --pooler_type avg \
    "$@"
```

Noted that you should prepare the T5 checkpoints and download T0's training and evaluation task data mentioned in paper to run through the above code.

Some part of our code is adapted from [SimCSE](https://github.com/princeton-nlp/SimCSE)