python eval.py \
    --eval 4096 \
    --t_scale 0.05 --T 1 --epochs 10000 \
    --batch_size 6 --energy xtb --local_model "weights/egnn_vacuum_small_with_hs_final" \
    --patience 25000 --model mlp \
    --smiles "CCCCCC(=O)OC" --log_var_range 0.5 \
    --temperature 300 --zero_init --clipping --pis_architectures \
    --mode_fwd tb-avg --lr_policy 1e-4 --lr_back 1e-4 --lr_flow 1e-3 \
    --hidden_dim 512 --joint_layers 5 --s_emb_dim 512 --t_emb_dim 512 --harmonics_dim 512

python eval.py \
    --eval 4096 \
    --t_scale 0.25 --T 5 --epochs 10000 \
    --batch_size 6 --energy xtb --local_model "weights/egnn_solvation_small_with_hs_final" \
    --patience 25000 --model mlp \
    --smiles "CCCCCC(=O)OC" \
    --temperature 300 --zero_init --clipping --pis_architectures \
    --mode_fwd tb-avg --lr_policy 1e-4 --lr_back 1e-4 --lr_flow 1e-3 \
    --hidden_dim 512 --joint_layers 5 --s_emb_dim 512 --t_emb_dim 512 --harmonics_dim 512 \
    --solvate


