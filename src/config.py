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
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = "../data/"
    train_images = "../data/specs/train/"
    valid_images = "../data/specs/valid/"
    train_path = "../data/train.csv"
    valid_path = "../data/valid.csv"
    SR = 32000
    DURATION = 5
    MAX_READ_SAMPLES = 5
    LR = 5e-4
