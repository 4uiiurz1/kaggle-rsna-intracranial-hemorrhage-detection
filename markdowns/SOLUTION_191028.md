## Preprocessing
- Appian's windowing (c40w80, c80w200, c40w380)

## Augmentation
```python
train_transform = Compose([
  transforms.Resize(512, 512),
  transforms.HorizontalFlip(),
  transforms.ShiftScaleRotate(
      shift_limit=0.0625,
      scale_limit=0.1,
      border_mode=cv2.BORDER_CONSTANT,
      value=0,
      p=0.5
  ),
  transforms.RandomCrop(410, 410) if args.random_crop else NoOp(),
  transforms.Normalize(mean=model.mean, std=model.std),
  ToTensor(),
])
```

## Models
- EfficientNet B1-B4
- SE-ResNeXt50_32x4d

## Train
- Loss: ```BCEWithLogitsLoss(weight=torch.Tensor([1., 1., 1., 1., 1., 2.]))```
- Optimizer: RAdam
- LR scheduler: CosineAnnealingLR (lr=1e-3 -> 1e-5)
- 5 epochs
- use all train data

## Postprocessing (-0.005 LB imporoved)
1. Concat prediction using 'StudyInstanceUID' of metadata.
2. Apply gaussian moving averaging.

## Score
inference (with hflip TTA) -> simple averaging -> postprocess
PublicLB: 0.062
