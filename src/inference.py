import gc
from pathlib import Path
import pandas as pd
import torch
import pytorch_lightning as pl
import warnings
from models import BirdClefModel
from config import test_config as Config
from dataset import BirdDatasetTest

warnings.filterwarnings("ignore")
pl.seed_everything(Config.seed, workers=True)


def predict(data_loader, model):
    model.to("cpu")
    model.eval()
    predictions = []
    for en in range(len(ds_test)):
        print(en)
        images = torch.from_numpy(ds_test[en])
        print(images.shape)
        with torch.no_grad():
            outputs = model(images).sigmoid().detach().cpu().numpy()
            print(outputs.shape)
        #             pred_batch.extend(outputs.detach().cpu().numpy())
        #         pred_batch = np.vstack(pred_batch)
        predictions.append(outputs)

    return predictions


df_train = pd.read_csv(Config.train_path)
Config.num_classes = len(df_train.primary_label.unique())

df_test = pd.DataFrame(
    [
        (path.stem, *path.stem.split("_"), path)
        for path in Path(Config.test_path).glob("*.ogg")
    ],
    columns=["filename", "name", "id", "path"],
)
print(f"df_test shape: {df_test.shape}")
print(df_test.head())


print("Create Dataloader...")
ds_test = BirdDatasetTest(
    df_test,
    sr=Config.SR,
    duration=Config.DURATION,
)


audio_model = BirdClefModel()
print("Model Creation")
model = BirdClefModel.load_from_checkpoint(
    Config.model_ckpt, train_dataloader=None, validation_dataloader=None
)
print("Running Inference..")

preds = predict(ds_test, model)

gc.collect()
torch.cuda.empty_cache()
filenames = df_test.filename.values.tolist()

bird_cols = list(pd.get_dummies(df_train["primary_label"]).columns)
sub_df = pd.DataFrame(columns=["row_id"] + bird_cols)
print(f"sub_df shape: {sub_df.shape}")

for i, file in enumerate(filenames):
    pred = preds[i]
    num_rows = len(pred)
    row_ids = [f"{file}_{(i+1)*5}" for i in range(num_rows)]
    df = pd.DataFrame(columns=["row_id"] + bird_cols)

    df["row_id"] = row_ids
    df[bird_cols] = pred

    sub_df = pd.concat([sub_df, df]).reset_index(drop=True)

print(f"sub_df shape: {sub_df.shape}")
print(sub_df.head())
