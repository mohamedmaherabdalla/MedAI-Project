# Experiment log

| Date | Who | Config | Commit | Result | Output path |
|---|---|---|---|---|---|
| 2026-10-14 | Mohamed Abdalla | configs/unet_c1-c4_seed0.yaml | 9198d6b | U-Net (smp ResNet-34), seed 0, 100 epochs: C5 Dice 0.4668, IoU 0.4081 (Slurm 229355, 17 min) | /l/users/mohamed.abdalla/medai/outputs/unet_c1-c4_seed0 |
| 2026-10-14 | Mohamed Abdalla | configs/pranet_c1-c4_seed0.yaml | 9198d6b | PraNet (MyTrain.py defaults), seed 0, 19 epochs: C5 Dice 0.5796, IoU 0.5104 (Slurm 229356, 11 min) | /l/users/mohamed.abdalla/medai/outputs/pranet_c1-c4_seed0 |
| 2026-10-14 | Mohamed Abdalla | configs/polyp-pvt_c1-c4_seed0.yaml | 8606a8f | Polyp-PVT (Train.py defaults, last epoch), seed 0, 99 epochs: C5 Dice 0.6442, IoU 0.5721 (Slurm 229357, 67 min) | /l/users/mohamed.abdalla/medai/outputs/polyp-pvt_c1-c4_seed0 |
| 2026-10-14 | Mohamed Abdalla | configs/sam2-unet_c1-c4_seed0.yaml | 15bffa0 | SAM2-UNet (train.py defaults, Hiera-L), seed 0, 20 epochs: C5 Dice 0.6584, IoU 0.5930 (Slurm 229523, 0.35 h) | /l/users/mohamed.abdalla/medai/outputs/sam2-unet_c1-c4_seed0 |
| 2026-10-14 | Mohamed Abdalla | configs/emcad_c1-c4_seed0.yaml | 145673b | EMCAD (train_polyp.py defaults, one run, last epoch), seed 0, 200 epochs: C5 Dice 0.6608, IoU 0.5754 (Slurm 229524, 2.52 h) | /l/users/mohamed.abdalla/medai/outputs/emcad_c1-c4_seed0 |
| 2026-10-14 | Mohamed Abdalla | configs/nnunet-2d_c1-c4_seed0.yaml | 99fa3d4 | nnU-Net v2 (2d, fold all, default plans and trainer), seed 0, 1000 epochs: C5 Dice 0.5938, IoU 0.5134 (Slurm 229538, 12.05 h; resume job 229539 found checkpoint_final and only re-predicted C5, 2 min) | /l/users/mohamed.abdalla/medai/outputs/nnunet-2d_c1-c4_seed0 |
