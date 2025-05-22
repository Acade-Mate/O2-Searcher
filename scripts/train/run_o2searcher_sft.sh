set -x
export WANDB_API_KEY="" # replace wandb api key

nproc_per_node=$1

save_path="./checkpoints/o2searcher/coldstart/"
model_path="Qwen2.5-3B-Instruct"
train_files="./o2searcher/data/coldstart/train.json"
val_files="./o2searcher/data/coldstart/test.json"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.train_batch_size=16 \
    data.micro_batch_size=4 \
    data.max_length=10240 \
    model.partial_pretrain=$model_path \
    optim.lr=1e-5 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=o2searcher \
    trainer.experiment_name=qwen2.5-3b-sft \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=2