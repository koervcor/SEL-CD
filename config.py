
MSTAR = {
  "epoch": 100,
  "num_classes": 10,
  "train_image_path": "./data/mstar/TRAIN/17_DEG/",
  "valid_image_path": "./data/mstar/VAL/17_DEG/",
  "test_image_path": "./data/mstar/TEST/left/15_DEG_50/",
  "predict_image_path": "./data/mstar/TEST/",
  "lr": 0.01,
  "gamma": 0.5,
  "step_lr": [10, 30, 60, 80],
  "image_format": "jpeg",
  "model_output_dir": "./chkpt",
  "chkpt": "MSTAR.pth",
  "predict_model": "./chkpt/MSTAR.pth",
  "mean": [0.37918335],
  "std": [0.20051193]
}
