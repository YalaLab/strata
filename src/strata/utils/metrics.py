from sklearn.metrics import f1_score, precision_score, recall_score


def F1(gt_column, pred_column, inference_results_df):
    # Exclude nan items in F1 score calculation
    keep_if_labeled = ~inference_results_df[gt_column].isna()
    gt = inference_results_df[gt_column][keep_if_labeled]
    pred = inference_results_df[pred_column][keep_if_labeled]
    if len(set(gt)) == 2:
        average = "binary"
    else:
        average = "macro"
    return f1_score(y_true=gt, y_pred=pred, average=average) * 100


def Precision(gt_column, pred_column, inference_results_df):
    # Exclude nan items in precision score calculation
    keep_if_labeled = ~inference_results_df[gt_column].isna()
    gt = inference_results_df[gt_column][keep_if_labeled]
    pred = inference_results_df[pred_column][keep_if_labeled]
    if len(set(gt)) == 2:
        average = "binary"
    else:
        average = "macro"
    return precision_score(y_true=gt, y_pred=pred, average=average) * 100


def Recall(gt_column, pred_column, inference_results_df):
    # Exclude nan items in recall score calculation
    keep_if_labeled = ~inference_results_df[gt_column].isna()
    gt = inference_results_df[gt_column][keep_if_labeled]
    pred = inference_results_df[pred_column][keep_if_labeled]
    if len(set(gt)) == 2:
        average = "binary"
    else:
        average = "macro"
    return recall_score(y_true=gt, y_pred=pred, average=average) * 100
