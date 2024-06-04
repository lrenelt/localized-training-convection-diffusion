# -*- tab-width: 4; indent-tabs-mode: nil  -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    filename = sys.argv[1]
    print(filename)
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        basisSize = (len(header)-1)//4
        errors = []
        for row in reader:
            errors.append(np.array(row[1:basisSize], dtype=float))

        ntest = len(errors)
        errors = np.array(errors)

        maxErrors = np.max(errors, axis=0)
        minErrors = np.min(errors, axis=0)

        xvals = np.arange(1,basisSize)

        plt.figure()
        plt.plot(xvals, maxErrors, label='max error over 20 test solutions')
        plt.plot(xvals, minErrors, label='min error over 20 test solutions')
        plt.yscale('log')
        plt.legend()
        plt.title('Decay of the approximation error')
        plt.show()
