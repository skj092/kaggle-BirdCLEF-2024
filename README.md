# BirdCLEF 2024

## To Identify Bird Species from their Sounds

## Provided Data
* **train_audios**: Contain 182 folders, each folder contain multiple audio files of particular bird species.
* **train_metadata.csv**: Contain metadata of each audio file.
* **unlabeled_soundscapes**: Contain multiple audio files of bird sounds.
* **eBIrd_Taxonomy_v2021.csv**: Data on the relationships between different species.
* **test_soundscapes**: Only the hidden rerun copy of the test soundscape dirctory will only be populated.
* **SampleSubmission.csv**: A sample submission file in the correct format. First column is the filename, and remaining are scores for each bird species.

## Evaluation Metric:
* A version of macro-averaged ROC-AUC that skips classes which have no true positive labels.


## Pipeline

### 1. Convert the audio to spectrogram and train a simple image classification model.

## Resources:
1. Convert Audio To Spectrogram: https://www.kaggle.com/code/nischaydnk/split-creating-melspecs-stage-1
2. Pytorch Lightning Inferece: https://www.kaggle.com/code/nischaydnk/birdclef-2023-pytorch-lightning-inference
3. Pytorch LIghtning training: https://www.kaggle.com/code/nischaydnk/birdclef-2023-pytorch-lightning-training-w-cmap

2. Pytorch Lightning Inference

1. import necessary libraries
2. Define configs
3. define functions:
    * computer melspec
    * mono_to_clolor
    * crop or pad
4. read train csv file
5. Update config num_classes wth totoal number of unique primary labels
6. Create test dataframe with filename, name, id ,path columns
7. def transform for augmentation
8. Define dataset class


---------------------------------
1. preparing dataset
1. read the audio file with soundfile module
