import torch

def metrics_from_cm(cm):
    TP = cm.diag()
    FP = cm.sum(0) - TP
    FN = cm.sum(1) - TP
    TN = torch.empty(cm.shape[0], device=cm.device)
    for i in range(cm.shape[0]):
        TN[i] = cm[:i,:i].sum() + cm[:i,i:].sum() + cm[i:,:i].sum() + cm[i:,i:].sum()

    precision = TP/(TP+FP)
    recall = TP/(TP+FN) # this is the same as accuracy per class
    f1 = 2*(precision*recall)/(precision + recall)
    iou = TP/(TP+FP+FN) # iou per class
    accuracy = (TP + TN)/(TP + FP + FN + TN)

    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "accuracy": accuracy}

def print_cm(cm, num_classes):
    cm=cm/cm.sum(1, keepdim=True)
    print("       ", "".join([f"Pr {i:>2} " for i in range(num_classes)]))
    print("-------------------------------------------------------------------------------------------------------")
    for i in range(num_classes):
        print(f"True {i:>2}|", end="")
        for j in range(num_classes):
            print(f"{cm[i, j]:>5.3f} ".replace("0.", " ."), end="")
        print()

def group_classes(cm, groups):
    '''
    Given a CM of k classes, and a list of lists containing the groups of labels
    such as [[0, 1]] i.e. consider the label 0 equal to the label 2 computes:

    CM' os size k - (sum(len(group)) in groups) + num_groups

    Labels that are not grouped are kept at the same original order, and grouped
    labels go to the end, that is, for the example group [0, 1], and 4 labels
    (0, 1, 2, 3), the new set of labels would be: 0->2, 1->3, 2->[0, 1]

    In cm' the new values are computed by summing the original values according to
    the new labels
    '''

    groups_all = [x for xs in groups for x in xs]
    n_groups_all = [i for i in range(cm.shape[0]) if i not in groups_all]

    k = cm.shape[0] - len(groups_all)
    new_size = k + len(groups)
    new_cm = torch.zeros((new_size, new_size), dtype=torch.int64)

    new_cm[:k, :k] = cm[n_groups_all][:, n_groups_all]

    for i, g in enumerate(groups):
        new_cm[k+i, k+i] = cm[groups[i]][:, groups[i]].sum()
        new_cm[:k, k+i] = cm[n_groups_all][:, groups[i]].sum(1)
        new_cm[k+i, :k] = cm.T[n_groups_all][:, groups[i]].sum(1)

        for j in range(len(groups)):
            if i!=j:
                new_cm[k+i, k+j] = cm[groups[i]][:, groups[j]].sum()

    return new_cm