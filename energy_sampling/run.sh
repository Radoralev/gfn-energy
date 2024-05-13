export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python train.py --t_scale 1. \
    --batch_size 32 \
    --energy xtb \
    --smiles "CCCCCC(=O)OC" \
    --temperature 300 \
    --solvate \
    --pis_architectures \
    --zero_init --clipping \
    --mode_fwd tb --mode_bwd tb \
    --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1 \
    --exploratory --exploration_wd --exploration_factor 0.1  \
    --buffer_size 60000 --prioritized rank --rank_weight 0.01 \
    --ld_step 0.1 --ld_schedule \
    --target_acceptance_rate 0.574 \
    --hidden_dim 256 --joint_layers 5 --s_emb_dim 256 --t_emb_dim 256 --harmonics_dim 256 
