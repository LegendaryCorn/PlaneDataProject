####################################################################
# main.py
####################################################################
# main is used to execute the code and run experiments.
####################################################################

import numpy as np

def main():
    try:
        data = np.load("Data/DASHlink_full_fourclass_raw_comp.npz")
    except:
        print("Data/DASHlink_full_fourclass_raw_comp.npz not found! Please download the file from https://c3.ndc.nasa.gov/dashlink/resources/1018/")
        return
    # x = data['data'] | 99837 samples, 20 features over 160 seconds (1 second sampling rate), technically 3200 features total
    # y = data['label'] | 99837 labels

    # The data is sorted by labels
    # [89663, 7013, 2207, 954]

    y = data['label']
    ls = [0, 0, 0, 0]
    for label in y:
        ls[int(label)] += 1

    print(ls)

if __name__ == "__main__":
    main()