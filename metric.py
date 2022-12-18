from itertools import product

import numpy as np


def true_positives(y_true: np.ndarray ,y_predictions: np.ndarray, true_lbl:int) -> np.ndarray:
    return np.logical_and(y_true==true_lbl ,y_predictions == true_lbl)

def true_negatives(y_true: np.ndarray ,y_predictions: np.ndarray, true_lbl:int) -> np.ndarray:
    return np.logical_and(y_true!=true_lbl ,y_predictions != true_lbl)

def false_positives(y_true: np.ndarray ,y_predictions: np.ndarray, true_lbl:int) -> np.ndarray:
    return np.logical_and(y_true!=true_lbl ,y_predictions == true_lbl)

def false_negatives(y_true: np.ndarray ,y_predictions: np.ndarray, true_lbl:int) -> np.ndarray:
    return np.logical_and(y_true==true_lbl ,y_predictions != true_lbl)    

def precision(y_true: np.ndarray ,y_predictions: np.ndarray, true_lbl:int) -> float:
    TP = true_positives(y_true,y_predictions, true_lbl)
    FP = false_positives(y_true,y_predictions, true_lbl)
    TP,FP=np.sum(TP),np.sum(FP)
    return TP/(TP + FP)

def recall(y_true: np.ndarray ,y_predictions: np.ndarray, true_lbl:int) -> float:
    TP = true_positives(y_true,y_predictions, true_lbl)
    FN = false_negatives(y_true,y_predictions, true_lbl)
    TP,FN=np.sum(TP),np.sum(FN)
    return TP/(TP + FN)

def F1_score(y_true: np.ndarray ,y_predictions: np.ndarray, true_lbl:int) -> float:
    precision_ = precision(y_true,y_predictions,true_lbl)
    recall_ = recall(y_true,y_predictions,true_lbl)
    return 2*(precision_*recall_)/(precision_+recall_)

def accuracy(y_true: np.ndarray ,y_predictions: np.ndarray) -> float:
    correct_pred = np.sum(np.equal(y_true,y_predictions))
    total_observation = y_true.shape[0]
    return correct_pred/total_observation

def confusion_matrxi(y_true: np.ndarray ,y_predictions:np.ndarray) -> np.ndarray:
    unique_class = np.unique(np.concatenate((y_true,y_predictions)))
    n_class = unique_class.shape[0]
    zipped_list = list(zip(y_true,y_predictions))
    confu_mat = []
    for i in product(unique_class,repeat=2):
        confu_mat.append(zipped_list.count(i))
    confu_mat= np.asarray(confu_mat).reshape(n_class,n_class)
    return confu_mat    


