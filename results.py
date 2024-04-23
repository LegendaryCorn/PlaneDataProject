####################################################################
# results.py
####################################################################
# Processes the results; can print it out using matplotlib.
####################################################################

def accuracy(y, pred, acc):
    count, total = 0,0
    for res in acc:
        if res:
            count += 1
        total += 1
    print(count, '/', total)
    print(count/total * 100,'%')