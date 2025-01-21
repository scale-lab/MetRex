import numpy as np 
from utils import parse_metrics


def compute_mre(
    pred: int, 
    gold: int
):
    if pred is None:
       mre = 1
    else: 
        mre = np.round(np.abs(pred - gold) / gold, 2)
    return mre


def compute_acc_at_k_majority(
    answers: dict, 
    ground_truth: list, 
    k: list =[1, 5, 10], 
    tolerance: float =0.2, 
    function=np.median
):
    """
    Compute acc@k using majority voting
    
    Args:
        answers: Dictionary of predictions by metric
        ground_truth: List of ground truth values
        k: List of k values to evaluate
        tolerance: MRE tolerance threshold
        function: Function to compute majority (default: np.median)
    
    Returns:
        List of dictionaries containing acc@k values for each metric and k value
    """
    acc_at_k = []
    for k_value in k:
        acc_at_k_value = dict()
        for metric in answers.keys():
            if len(answers[metric]) == 0:
                acc_at_k_value[metric] = 0
                continue
            correct_count = 0
            total_problems = len(answers[metric])
            
            # Process each problem
            for i, answer in enumerate(answers[metric]):
                correct = count_majority(
                    answers=answer,
                    ground_truth=ground_truth[i][metric],
                    metric=metric,
                    tolerance=tolerance,
                    k=k_value,
                    function=function
                )
                correct_count += correct
            
            # Calculate accuracy as percentage of correct majority predictions
            acc_at_k_value[metric] = correct_count / total_problems if total_problems > 0 else 0
        
        acc_at_k.append(acc_at_k_value)
    
    return acc_at_k


def count_majority(
    answers: list, 
    ground_truth: float, 
    metric: str, 
    tolerance: float, 
    k: int, 
    function=np.median
):
    """
    Count if the majority (using specified function) of predictions is within tolerance
    
    Args:
        answers: List of predictions
        ground_truth: True value
        metric: Metric being evaluated
        tolerance: MRE tolerance threshold
        k: Number of predictions to consider (if None, use all)
        function: Function to compute majority (default: np.median)
    
    Returns:
        1 if majority prediction is within tolerance, 0 otherwise
    """
    # Get all valid predictions
    predictions = []
    for sample in answers:
        pred_parsed = parse_metrics({metric: sample}, metrics=[metric])
        if pred_parsed[metric] is not None:
            predictions.append(pred_parsed[metric])
    
    if not predictions:
        return 0
        
    # Take first k predictions if specified
    predictions = predictions[:k]
    
    majority_pred = function(predictions)

    # Compute MRE for majority prediction
    mre = compute_mre(majority_pred, ground_truth)

    return 1 if mre <= tolerance else 0
