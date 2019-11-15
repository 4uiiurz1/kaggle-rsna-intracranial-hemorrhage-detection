# kaggle-rsna-intracranial-hemorrhage-detection

1. Put the input files in `inputs/` and unzip as the following structure:
```
inputs
├── stage_1_train_images
│   ├── ...
|
├── stage_x_test_images
│   ├── ...
│
├── stage_1_train.csv
│
└── stage_x_sample_submission.csv
```

2. Preprocess.
```
python create_metadata.py
python create_windowing_images.py
```

3. Train the models.
```
python train.py --name efficientnet-b1_102407 --arch efficientnet-b1 --apex
python train.py --name efficientnet-b2_102000 --arch efficientnet-b2 --apex
python train.py --name efficientnet-b3_102112 --arch efficientnet-b3 --apex --img_size 480 --crop_size 384
python train.py --name efficientnet-b4_102100 --arch efficientnet-b4 --apex
python train.py --name se_resnext50_32x4d_102105 --arch se_resnext50_32x4d --apex
```

4. Test.
```
python test.py --name efficientnet-b1_102407 --hflip True
python test.py --name efficientnet-b2_102000 --hflip True
python test.py --name efficientnet-b3_102112 --hflip True
python test.py --name efficientnet-b4_102100 --hflip True
python test.py --name se_resnext50_32x4d_102105 --hflip True
```

4. Ensemble.
```
python average.py
```

5. Postprocess.
```
python gma.py --name average_102800
```
