#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_MODELSCOPE="0"
export WANDB_API_KEY="" # replace the wandab api key 

TRAIN_FILES="./o2searcher/data/hybrid/train.parquet"
VAL_FILES="./o2searcher/data/hybrid/test.parquet"
MODEL_PATH="./checkpoints/o2searcher/coldstart/global_step_306"

# Train over a single node, 4 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=32 \
    data.val_batch_size=256 \
    data.max_prompt_length=8096 \
    data.max_response_length=2048 \
    data.max_start_length=2048 \
    data.max_obs_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.state_masking=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n_agent=8 \
    actor_rollout_ref.rollout.n_agent_val=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='o2searcher' \
    trainer.experiment_name='qwen2.5-3b-grpo' \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=150 \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=151 \
    trainer.total_epochs=3 \
    agent.max_turns=4 \
    searcher.urls.openended="http://127.0.0.1:10102/search" \
    searcher.urls.closedended="http://127.0.0.1:10001/wiki_search" \
    searcher.topk=3 "${@:1}"