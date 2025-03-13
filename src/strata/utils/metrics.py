from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd

def map_to_categorical(gt, pred):
    """
    Maps ground truth (gt) and predictions (pred) to the same categorical integer codes,
    only if they are not already integers.
    Parameters
    ----------
    gt : pd.Series
        The ground truth values.
    pred : pd.Series
        The prediction values.
    Returns
    -------
    pd.Series, pd.Series
        The ground truth and predictions, either as-is (if integers) or mapped to integer codes.
    """
    # Check if gt and pred are already integers
    if pd.api.types.is_integer_dtype(gt) and pd.api.types.is_integer_dtype(pred):
        return gt, pred

    # Create a unified Categorical mapping based on both gt and pred
    categories = pd.Categorical(pd.concat([gt, pred], ignore_index=True))

    # Map gt and pred using the same categories
    gt_mapped = pd.Categorical(gt, categories=categories.categories).codes
    pred_mapped = pd.Categorical(pred, categories=categories.categories).codes

    return gt_mapped, pred_mapped


def F1(gt_column, pred_column, inference_results_df):
    """
    Computes the F1 score, with support for categorical ground truth and predictions.
    Ensures consistent mapping between ground truth and predictions, only if needed.
    Parameters
    ----------
    gt_column : str
        The column name for the ground truth values in the DataFrame.
    pred_column : str
        The column name for the predicted values in the DataFrame.
    inference_results_df : pd.DataFrame
        The DataFrame containing the ground truth and prediction columns.
    Returns
    -------
    float
        The F1 score as a percentage.
    """
    # Exclude NaN items in F1 score calculation
    keep_if_labeled = ~inference_results_df[gt_column].isna()
    gt = inference_results_df[gt_column][keep_if_labeled]
    pred = inference_results_df[pred_column][keep_if_labeled]
    # Map ground truth and predictions to categorical integer codes, if needed
    gt_mapped, pred_mapped = map_to_categorical(gt, pred)

    # Determine whether to use binary or macro average
    average = "binary" if len(set(gt_mapped)) == 2 else "macro"

    # Compute and return F1 score
    return f1_score(y_true=gt_mapped, y_pred=pred_mapped, average=average) * 100


def Precision(gt_column, pred_column, inference_results_df):
    """
    Computes the precision score, with support for categorical ground truth and predictions.
    Ensures consistent mapping between ground truth and predictions, only if needed.
    Parameters
    ----------
    gt_column : str
        The column name for the ground truth values in the DataFrame.
    pred_column : str
        The column name for the predicted values in the DataFrame.
    inference_results_df : pd.DataFrame
        The DataFrame containing the ground truth and prediction columns.
    Returns
    -------
    float
        The precision score as a percentage.
    """
    # Exclude NaN items in precision score calculation
    keep_if_labeled = ~inference_results_df[gt_column].isna()
    gt = inference_results_df[gt_column][keep_if_labeled]
    pred = inference_results_df[pred_column][keep_if_labeled]
    # Map ground truth and predictions to categorical integer codes, if needed
    gt_mapped, pred_mapped = map_to_categorical(gt, pred)

    # Determine whether to use binary or macro average
    average = "binary" if len(set(gt_mapped)) == 2 else "macro"

    # Compute and return precision score
    return precision_score(y_true=gt_mapped, y_pred=pred_mapped, average=average) * 100


def Recall(gt_column, pred_column, inference_results_df):
    """
    Computes the recall score, with support for categorical ground truth and predictions.
    Ensures consistent mapping between ground truth and predictions, only if needed.
    Parameters
    ----------
    gt_column : str
        The column name for the ground truth values in the DataFrame.
    pred_column : str
        The column name for the predicted values in the DataFrame.
    inference_results_df : pd.DataFrame
        The DataFrame containing the ground truth and prediction columns.
    Returns
    -------
    float
        The recall score as a percentage.
    """
    # Exclude NaN items in recall score calculation
    keep_if_labeled = ~inference_results_df[gt_column].isna()
    gt = inference_results_df[gt_column][keep_if_labeled]
    pred = inference_results_df[pred_column][keep_if_labeled]
    # Map ground truth and predictions to categorical integer codes, if needed
    gt_mapped, pred_mapped = map_to_categorical(gt, pred)

    # Determine whether to use binary or macro average
    average = "binary" if len(set(gt_mapped)) == 2 else "macro"

    # Compute and return recall score
    return recall_score(y_true=gt_mapped, y_pred=pred_mapped, average=average) * 100
