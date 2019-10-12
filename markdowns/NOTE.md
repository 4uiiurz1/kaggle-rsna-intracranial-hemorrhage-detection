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
| 092805 | o |         | o     | o     |        |          | 0.0853   | 0.0731 | 0.084 | -        |

## Architecture
- WeightedBCEWithLogitsLoss
- RAdam
- 5epochs
- Cosine Annealing (1e-3 -> 1e-5)
- hflip (p=0.5)
- shift (limit=0.0625, p=0.5)
- scale (limit=0.1, p=0.5)
- img_size: 256

| model_name             | val loss | PublicLB |
|:----------------------:|:--------:|:--------:|
| efficientnet-b0_100407 | 0.0783   | 0.080    |
| efficientnet-b1_100509 | 0.0772   | 0.081    |
| efficientnet-b2_100701 | 0.0776   | 0.078    |
| efficientnet-b3_100712 | 0.0766   | 0.078    |

## Crop
- WeightedBCEWithLogitsLoss
- RAdam
- 5epochs
- Cosine Annealing (1e-3 -> 1e-5)
- hflip (p=0.5)
- shift (limit=0.0625, p=0.5)
- scale (limit=0.1, p=0.5)

| model_name             | crop      | img_size | crop_size | val loss | PublicLB |
|:----------------------:|:---------:|:--------:|:---------:|:--------:|:--------:|
| efficientnet-b0_100822 | center    | 288      | 256       | 0.0799   | 0.080    |
| efficientnet-b0_100910 | center    | 320      | 256       | 0.0772   | 0.079    |
| efficientnet-b0_100923 | fg center | 320      | 256       | 0.0779   | 0.081    |
| efficientnet-b0_101022 | random    | 320      | 256       | 0.0783   | **0.078**|
| efficientnet-b0_101106 | random    | 352      | 256       | 0.0782   | 0.078    |

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

## EfficientNet
```py
params_dict = {
    # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
}
```
