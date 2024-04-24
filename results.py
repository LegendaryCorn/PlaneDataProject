####################################################################
# results.py
####################################################################
# Processes the results; can print it out using matplotlib.
####################################################################

def accuracy(y, pred, acc):

    total = len(y)
    fp, fn, sp, tp, tn = 0, 0, 0, 0, 0
    for i in range(total):
        tp += 1 if y[i] != 0 and pred[i] == y[i] else 0
        tn += 1 if y[i] == 0 and pred[i] == y[i] else 0
        fp += 1 if y[i] == 0 and pred[i] != 0 else 0
        fn += 1 if y[i] != 0 and pred[i] == 0 else 0
        sp += 1 if y[i] != 0 and pred[i] != 0 and pred[i] != y[i] else 0


    print("Total Accuracy:", tp + tn, '/', total, "|", (tp + tn)/total * 100,'%')
    print("Anomaly Prediction Accuracy:", tp, '/', (tp+sp+fn), "|", (tp)/(tp+sp+fn) * 100,'%')
    print("Anomaly Detection Accuracy:", (tp+sp), '/', (tp+sp+fn), "|", (tp+sp)/(tp+sp+fn) * 100,'%')
    print("Nonanomaly Detection Accuracy:", tn, '/', (tn+fp), "|", (tn)/(tn+fp) * 100,'%')