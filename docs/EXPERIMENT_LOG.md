# Experiment log

| Date | Who | Config | Commit | Result | Output path |
|---|---|---|---|---|---|
| 2026-10-14 | Mohamed Abdalla | configs/unet_c1-c4_seed0.yaml | 9198d6b | U-Net (smp ResNet-34), seed 0, 100 epochs: C5 Dice 0.4668, IoU 0.4081 (Slurm 229355, 17 min) | /l/users/mohamed.abdalla/medai/outputs/unet_c1-c4_seed0 |
| 2026-10-14 | Mohamed Abdalla | configs/pranet_c1-c4_seed0.yaml | 9198d6b | PraNet (MyTrain.py defaults), seed 0, 19 epochs: C5 Dice 0.5796, IoU 0.5104 (Slurm 229356, 11 min) | /l/users/mohamed.abdalla/medai/outputs/pranet_c1-c4_seed0 |
| 2026-10-14 | Mohamed Abdalla | configs/polyp-pvt_c1-c4_seed0.yaml | 8606a8f | Polyp-PVT (Train.py defaults, last epoch), seed 0, 99 epochs: C5 Dice 0.6442, IoU 0.5721 (Slurm 229357, 67 min) | /l/users/mohamed.abdalla/medai/outputs/polyp-pvt_c1-c4_seed0 |
