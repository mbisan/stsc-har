import torch

def metrics_from_cm(cm):
    TP = cm.diag()
    FP = cm.sum(0) - TP
    FN = cm.sum(1) - TP
    TN = torch.empty(cm.shape[0])
    for i in range(cm.shape[0]):
        TN[i] = cm[:i,:i].sum() + cm[:i,i:].sum() + cm[i:,:i].sum() + cm[i:,i:].sum()

    precision = TP/(TP+FP)
    recall = TP/(TP+FN) # this is the same as accuracy per class
    f1 = 2*(precision*recall)/(precision + recall)
    iou = TP/(TP+FP+FN) # iou per class

    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}