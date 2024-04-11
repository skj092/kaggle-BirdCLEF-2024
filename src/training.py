import gc
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import warnings
from config import Config
from dataset import get_fold_dls
from models import BirdClefModel
from utils import show_batch, padded_cmap
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

warnings.filterwarnings('ignore')


pl.seed_everything(Config.seed, workers=True)


def config_to_dict(cfg):
    return dict((name, getattr(cfg, name)) for name in dir(cfg) if not name.startswith('__'))


df_train = pd.read_csv(Config.train_path)
df_valid = pd.read_csv(Config.valid_path)
df_train.head()

Config.num_classes = len(df_train.primary_label.unique())

df_train = pd.concat(
    [df_train, pd.get_dummies(df_train['primary_label'])], axis=1)
df_valid = pd.concat(
    [df_valid, pd.get_dummies(df_valid['primary_label'])], axis=1)

# Take a subset
df_train = df_train.sample(n=100)
df_valid = df_valid.sample(n=100)

birds = list(df_train.primary_label.unique())
missing_birds = list(set(list(df_train.primary_label.unique())).difference(
    list(df_valid.primary_label.unique())))

non_missing_birds = list(
    set(list(df_train.primary_label.unique())).difference(missing_birds))

len(non_missing_birds)

df_valid[missing_birds] = 0
df_valid = df_valid[df_train.columns]  # Fix order


dl_train, dl_val, ds_train, ds_val = get_fold_dls(df_train, df_valid)
show_batch(ds_val, 8, 2, 4)


dummy = df_valid[birds].copy()
dummy[birds] = np.random.rand(dummy.shape[0], dummy.shape[1])

padded_cmap(df_valid[birds], dummy[birds], padding_factor=5)

padded_cmap(df_valid[birds], dummy[birds], padding_factor=1)

# map_score(df_valid[birds], dummy[birds])


def run_training():
    print("Running training...")
    logger = None

    dl_train, dl_val, ds_train, ds_val = get_fold_dls(df_train, df_valid)

    audio_model = BirdClefModel()

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=Config.PATIENCE, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath="exp1/",
                                          save_top_k=1,
                                          save_last=True,
                                          save_weights_only=True,
                                          filename=f'./{Config.model}_loss',
                                          verbose=True,
                                          mode='min')

    callbacks_to_use = [checkpoint_callback, early_stop_callback]

    trainer = pl.Trainer(
        # gpus=0,
        val_check_interval=0.5,
        deterministic=True,
        max_epochs=Config.epochs,
        logger=logger,
        # auto_lr_find=False,
        callbacks=callbacks_to_use,
        # precision=Config.PRECISION, accelerator="gpu"
    )

    print("Running trainer.fit")
    trainer.fit(audio_model, train_dataloaders=dl_train,
                val_dataloaders=dl_val)

    gc.collect()
    torch.cuda.empty_cache()


run_training()
