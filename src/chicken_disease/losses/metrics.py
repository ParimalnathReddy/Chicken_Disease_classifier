from sklearn.metrics import f1_score, recall_score


def compute_macro_f1(y_true, y_pred, class_names):
    return f1_score(y_true, y_pred, average="macro", labels=range(len(class_names)), zero_division=0)


def compute_per_class_f1_and_recall(y_true, y_pred, class_names):
    f1s = f1_score(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )
    recalls = recall_score(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )
    out = {}
    for i, name in enumerate(class_names):
        out[name] = {"f1": float(f1s[i]), "recall": float(recalls[i])}
    return out
