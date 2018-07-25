import numpy as np
import scipy.io as scio
import sys

if len(sys.argv) >= 2:
    filename = sys.argv[1]
    temp_load = np.load(filename)
    # print variables in npy file
    print(temp_load.item())

    # if system argument is larger than 2
    if len(sys.argv) == 3:
        # if second system argument is '-c'
        if sys.argv[2] == '-c':
            # save npy to mat file
            file_mat = filename.replace(".npy", "")
            scio.savemat(file_mat, temp_load.item())

