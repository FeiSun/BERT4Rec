# python gen_data.py

CKPT_DIR="/apsarapangu/disk1/ofey.sf/BERT4Rec"
# CKPT_DIR="/dump/1/ofey.sf/BERT4Rec"
dataset_name="beauty"
max_seq_length=50
max_predictions_per_seq=30
masked_lm_prob=0.6

dim=16
batch_size=256
num_train_steps=400000

mask_prob=1.0
prop_sliding_window=0.1
dupe_factor=10
pool_size=10

signature="-mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-df${dupe_factor}-mpps${max_predictions_per_seq}-msl${max_seq_length}-fin"


python -u gen_data_fin.py \
    --dataset_name=${dataset_name} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --mask_prob=${mask_prob} \
    --dupe_factor=${dupe_factor} \
    --masked_lm_prob=${masked_lm_prob} \
    --prop_sliding_window=${prop_sliding_window} \
    --signature=${signature} \
    --pool_size=${pool_size} \


CUDA_VISIBLE_DEVICES=1 python -u run_pretraining.py \
    --train_input_file=./data/${dataset_name}${signature}.train.tfrecord \
    --test_input_file=./data/${dataset_name}${signature}.test.tfrecord \
    --vocab_filename=./data/${dataset_name}${signature}.vocab \
    --user_history_filename=./data/${dataset_name}${signature}.his \
    --checkpointDir=${CKPT_DIR}/${dataset_name} \
    --signature=${signature}-${dim} \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=./bert_train/bert_config_${dataset_name}_${dim}.json \
    --batch_size=${batch_size} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=100 \
    --learning_rate=1e-4


# CUDA_VISIBLE_DEVICES=1 python -u run_pretraining.py \
#     --train_input_file=../data/ml-1m-s0.1-m0.15-d10-mm20-ms200.train.tfrecord \
#     --test_input_file=../data/ml-1m-s0.1-m0.15-d10-mm20-ms200.test.tfrecord \
#     --vocab_filename=../data/ml-1m-s0.1-m0.15-d10-mm20-ms200.vocab \
#     --user_history_filename=../data/ml-1m-s0.1-m0.15-d10-mm20-ms200.output \
#     --checkpointDir=${CKPT_DIR}/bert_train \
#     --init_checkpoint=${CKPT_DIR}/bert_train${signature} \
#     --signature=${signature} \
#     --do_train=False \
#     --do_eval=True \
#     --bert_config_file=./bert_train/bert_config.json \
#     --batch_size=${batch_size} \
#     --max_seq_length=${max_seq_length} \
#     --max_predictions_per_seq=${max_predictions_per_seq} \
#     --num_train_steps=${num_train_steps} \
#     --num_warmup_steps=10 \
#     --item_vocab_size=3416 \
#     --learning_rate=1e-4


#     --init_checkpoint=${CKPT_DIR}/bert_train \
