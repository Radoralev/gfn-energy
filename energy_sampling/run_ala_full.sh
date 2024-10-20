export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python train.py --t_scale 1. --T 10 \
    --batch_size 256 \
    --epochs 50000 \
    --energy alanine_vacuum_full \
    --local_model 'weights/egnn_vacuum_batch_size_32' \
   # --conditional_flow_model \
    --temperature 300 \
    --zero_init --clipping \
    --pis_architectures \
    --mode_fwd tb-avg \
    --mode_bwd tb-avg \
    --lr_policy 1e-5 --lr_back 1e-5 --lr_flow 1e-2 \
    --hidden_dim 512 --joint_layers 5 --s_emb_dim 512 --t_emb_dim 512 --harmonics_dim 512
   --exploratory --exploration_wd --exploration_factor 0.1  \
   --buffer_size 1000000 --prioritized rank --rank_weight 0.01 \
   --ld_step 0.1 --ld_schedule \
   --target_acceptance_rate 0.574 \

# python train.py --t_scale 1. --T 100 \
#     --batch_size 32 \
#     --energy neural \
#     --local_model 'weights/egnn_solvation_batch_size_32' \
#     --smiles "CCCCCC(=O)OC" \
#     --temperature 300 \
#     --zero_init --clipping \
#     --equivariant_architectures \
#     --mode_fwd tb \
#     --mode_bwd tb \
#     --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1 \
#     --exploratory --exploration_wd --exploration_factor 0.1  \
#     --buffer_size 60000 --prioritized rank --rank_weight 0.01 \
#     --ld_step 0.1 --ld_schedule \
#     --target_acceptance_rate 0.574 \
#     --hidden_dim 32 --joint_layers 5 --s_emb_dim 32 --t_emb_dim 32 --harmonics_dim 32 
