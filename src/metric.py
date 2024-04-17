import pandas as pd
import pandas.api.types

import kaggle_metric_utilities

import sklearn.metrics


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Version of macro-averaged ROC-AUC score that ignores all classes that have no true positive labels.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {
            x: submission[x].dtype for x in submission.columns if not pandas.api.types.is_numeric_dtype(submission[x])}
        raise ParticipantVisibleError(
            f'Invalid submission data types found: {bad_dtypes}')

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0

    return kaggle_metric_utilities.safe_call_score(sklearn.metrics.roc_auc_score, solution[scored_columns].values, submission[scored_columns].values, average='macro')


def test_score():
    solution = pd.DataFrame({
        'row_id': [1, 2, 3],
        'class1': [0, 1, 0],
        'class2': [0, 0, 1],
        'class3': [1, 0, 0],
    })
    submission = pd.DataFrame({
        'row_id': [1, 2, 3],
        'class1': [0.1, 0.2, 0.3],
        'class2': [0.4, 0.5, 0.6],
        'class3': [0.7, 0.8, 0.9],
    })
    assert score(solution, submission, 'row_id') == 0.5
    print('test_score passed')


if __name__ == '__main__':
    test_score()
    data_dir = "/home/sonu/personal/kaggle-BirdCLEF/data"
    submission = pd.read_csv(f'{data_dir}/sample_submission.csv')
    submission1 = pd.read_csv(f'{data_dir}/sample_submission.csv')
    '''
    format of submission file
    'row_id', 'class1', 'class2', 'class3'
    'soundfile1', 0.1, 0.2, 0.3
    '''
    print(score(submission, submission1, 'row_id'))
