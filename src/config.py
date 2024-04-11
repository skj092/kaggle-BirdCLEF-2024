import torch


class Config:
    use_aug = False
    num_classes = 182
    batch_size = 64
    epochs = 12
    PRECISION = 16
    PATIENCE = 8
    seed = 2023
    model = "tf_efficientnet_b0_ns"
    pretrained = True
    weight_decay = 1e-3
    use_mixup = True
    mixup_alpha = 0.2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "/home/sonu/personal/BirdCLEF/data"
    train_images = data_root + "/specs/train/"
    valid_images = data_root + "/specs/valid/"
    train_path = data_root + "/train.csv"
    valid_path = data_root + "/valid.csv"
    SR = 32000
    DURATION = 5
    MAX_READ_SAMPLES = 5
    LR = 5e-4


class test_config:
    num_classes = 264
    batch_size = 12
    PRECISION = 16
    seed = 2023
    model = "tf_efficientnet_b0_ns"
    pretrained = False
    use_mixup = False
    mixup_alpha = 0.2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = "/home/sonu/personal/BirdCLEF/data"
    train_images = data_root + "/specs/train/"
    valid_images = data_root + "/specs/valid/"
    train_path = data_root + "/train.csv"
    valid_path = data_root + "/valid.csv"

    test_path = data_root + "/test_soundscapes/"
    SR = 32000
    DURATION = 5
    LR = 5e-4

    model_ckpt = "/home/sonu/personal/BirdCLEF/src/exp1/last.ckpt"
