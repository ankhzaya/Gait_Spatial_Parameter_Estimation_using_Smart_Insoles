import sys
import os
import numpy as np

sys.path.append('../')

def Accuracy(target, pred):
    correct = 0
    len_dataset = len(target)

    for i in range(len_dataset):
        if pred[i] == target[i]:
            correct += 1
        else:
            continue

    acc = 100. * correct / len_dataset
    return acc


def Edit_Distance(pred, target, norm=False):
    m_row = len(pred)
    n_col = len(target)
    D = np.zeros([m_row + 1, n_col + 1], np.float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if target[j - 1] == pred[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score

def segment_labels(Yi):
	idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
	Yi_split = np.array(([Yi[(idxs[i])] for i in range(len(idxs)-1)]))
	return Yi_split

def segment_data(Xi, Yi):
	idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
	Xi_split = [np.squeeze(Xi[:,idxs[i]:idxs[i+1]]) for i in range(len(idxs)-1)]
	Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs)-1)])
	return Xi_split, Yi_split

def segment_intervals(Yi):
	idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
	intervals = [(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)]
	return intervals

def segment_lengths(Yi):
	idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
	intervals = [(idxs[i+1]-idxs[i]) for i in range(len(idxs)-1)]
	return np.array(intervals)


def edit_score(P, Y, norm=True, bg_class=None, **kwargs):
    if type(P) == list:
        tmp = [edit_score(P[i], Y[i], norm, bg_class) for i in range(len(P))]
        return np.mean(tmp)
    else:
        P_ = segment_labels(P)
        Y_ = segment_labels(Y)
        if bg_class is not None:
            P_ = [c for c in P_ if c!=bg_class]
            Y_ = [c for c in Y_ if c!=bg_class]
        return Edit_Distance(P_, Y_, norm)


def overlap_f1(P, Y, n_classes=0, bg_class=None, overlap=.1, **kwargs):
    def overlap_(p, y, n_classes, bg_class, overlap):

        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        # Remove background labels
        if bg_class is not None:
            true_intervals = true_intervals[true_labels != bg_class]
            true_labels = true_labels[true_labels != bg_class]
            pred_intervals = pred_intervals[pred_labels != bg_class]
            pred_labels = pred_labels[pred_labels != bg_class]

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]

        # We keep track of the per-class TPs, and FPs.
        # In the end we just sum over them though.
        TP = np.zeros(n_classes, np.float)
        FP = np.zeros(n_classes, np.float)
        true_used = np.zeros(n_true, np.float)

        for j in range(n_pred):
            # Compute IoU against all others
            intersection = np.minimum(pred_intervals[j, 1], true_intervals[:, 1]) - np.maximum(pred_intervals[j, 0],
                                                                                               true_intervals[:, 0])
            union = np.maximum(pred_intervals[j, 1], true_intervals[:, 1]) - np.minimum(pred_intervals[j, 0],
                                                                                        true_intervals[:, 0])
            IoU = (intersection / union) * (pred_labels[j] == true_labels)

            # Get the best scoring segment
            idx = IoU.argmax()

            # If the IoU is high enough and the true segment isn't already used
            # Then it is a true positive. Otherwise is it a false positive.
            if IoU[idx] >= overlap and not true_used[idx]:
                TP[int(pred_labels[j])] += 1
                true_used[idx] = 1
            else:
                FP[int(pred_labels[j])] += 1

        TP = TP.sum()
        FP = FP.sum()
        # False negatives are any unused true segment (i.e. "miss")
        FN = n_true - true_used.sum()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (precision * recall) / (precision + recall)

        # If the prec+recall=0, it is a NaN. Set these to 0.
        F1 = np.nan_to_num(F1)

        return F1 * 100

    if type(P) == list:
        return np.mean([overlap_(P[i], Y[i], n_classes, bg_class, overlap) for i in range(len(P))])
    else:
        return overlap_(P, Y, n_classes, bg_class, overlap)


def overlap_score(P, Y, bg_class=None, **kwargs):
    # From ICRA paper:
    # Learning Convolutional Action Primitives for Fine-grained Action Recognition
    # Colin Lea, Rene Vidal, Greg Hager
    # ICRA 2016

    def overlap_(p, y, bg_class):
        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        if bg_class is not None:
            true_intervals = np.array([t for t, l in zip(true_intervals, true_labels) if l != bg_class])
            true_labels = np.array([l for l in true_labels if l != bg_class])
            pred_intervals = np.array([t for t, l in zip(pred_intervals, pred_labels) if l != bg_class])
            pred_labels = np.array([l for l in pred_labels if l != bg_class])

        n_true_segs = true_labels.shape[0]
        n_pred_segs = pred_labels.shape[0]
        seg_scores = np.zeros(n_true_segs, np.float)

        for i in range(n_true_segs):
            for j in range(n_pred_segs):
                if true_labels[i] == pred_labels[j]:
                    intersection = min(pred_intervals[j][1], true_intervals[i][1]) - max(pred_intervals[j][0],
                                                                                         true_intervals[i][0])
                    union = max(pred_intervals[j][1], true_intervals[i][1]) - min(pred_intervals[j][0],
                                                                                  true_intervals[i][0])
                    score_ = float(intersection) / union
                    seg_scores[i] = max(seg_scores[i], score_)

        return seg_scores.mean() * 100

    if type(P) == list:
        return np.mean([overlap_(P[i], Y[i], bg_class) for i in range(len(P))])
    else:
        return overlap_(P, Y, bg_class)
