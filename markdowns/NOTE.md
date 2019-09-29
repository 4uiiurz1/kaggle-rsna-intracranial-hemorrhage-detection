# Experiments
## Loss
- se_resnext50_32x4d
- freeze_bn
- RAdam
- 5epochs
- Cosine Annealing (1e-3 -> 1e-5)
- no augmentation

| Loss  | val loss | PublicLB |
|:-----:|:--------:|:---------:|
| BCEWithLogitsLoss | 0.0608 | 0.103 |
| WeightedBCEWithLogitsLoss (w=[1., 1., 1., 1., 1., 2.]) | 0.0794 | **0.090** |

## Augmentation
- resnet34
- freeze_bn
- WeightedBCEWithLogitsLoss
- RAdam
- 5epochs
- Cosine Annealing (1e-3 -> 1e-5)
- hflip (p=0.5)
- vflip (p=0.5)
- shift (limit=0.0625, p=0.5)
- scale (limit=0.1, p=0.5)
- rotate (limit=45, p=0.5)
- contrast (limit=0.2, p=0.5)

| model_name | hflip  | vflip | shift | scale | rotate | contrast | val loss | val score | PublicLB |
|:-----:|:------:|:-------:|:-----:|:-----:|:------:|:--------:|:--------:|:----------|:--------:|
| 092406 |   |    |    |       |        |          | 0.0883   | -    | 0.105 |
| 092412 | o |         |       |       |        |          | 0.0857   | -    | - |
| 092604 | o | o       |       |       |        |          | 0.0892   | -    | - |
| 092422 | o |         | o     |       |        |          | 0.0848   | -    | 0.085 |
| 092504 | o |         |       | o     |        |          | 0.0851   | -    | -        |
| 092510 | o |         |       |       | o      |          | 0.0854   | -    | -        |
| 092517 | o |         |       |       |        | o        | 0.0856   | -   | -        |
| 092611 | o |         | o     | o     | o      | o        | 0.0862   | -    | -        |
| 092805 | o |         | o     | o     |        | o        | 0.0853   | 0.0731 | 0.084 | -        |

## Other models
### se_resnext50_32x4d_092623
- freeze_bn
- WeightedBCEWithLogitsLoss
- 10epochs
- batch_size=32
- img_size=256
- RAdam (lr=1e-3, weight_decay=1e-4)
- CosineAnnealingLR (1e-3 -> 1e-5)
- hflip
- shift_scale_rotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=0, p=0.5)
- cv: False
- n_splits: 5

| val loss | PublicLB |
|:--------:|:--------:|
|  0.0742  |  0.082   |
