# Experiments
## Loss
- se_resnext50_32x4d
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

| hflip  | vflip   | shift | scale | rotate | contrast | val loss | PublicLB |
|:------:|:-------:|:-----:|:-----:|:------:|:--------:|:--------:|:--------:|
|        |         |       |       |        |          | 0.0883   | -        |
| o      |         |       |       |        |          | 0.0857   | -        |
| o      | o       |       |       |        |          | 0.0892   |          |
| o      |         | o     |       |        |          | 0.0848   |          |
| o      |         |       | o     |        |          | 0.0851   |          |
| o      |         |       |       | o      |          | 0.0854   |          |
| o      |         |       |       |        | o        | 0.0856   |          |
| o      |         | o     | o     | o      | o        | 0.0862   |   |
