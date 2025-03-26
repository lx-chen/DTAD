

## Run

#### Step 1. Setup the Anomaly Detection Dataset
Download the Anomaly Detection Dataset and convert it to MVTec AD format. 
The dataset folder structure should look like:
```
DATA_PATH/
    subset_1/
        train/
            good/
        test/
            good/
            defect_class_1/
            defect_class_2/
            defect_class_3/
            ...
    ...
```

#### Step 2. Running DTAD
```bash
python train.py
```
Parameters to be modified
- `dataset_root` denotes the path of the dataset.
- `classname` denotes the subset name of the dataset.
- `dataset` the dataset to train, which default='mvtecad'. 

Adjustable parameters for enhancing performance
- `batch_size` 
- `meta_epochs` 
- `sub_epochs` 
- `pos_beta` 
- `margin_tau` 
- `ramdn_seed` 
