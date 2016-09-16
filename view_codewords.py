from __future__ import print_function

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.show()

def view_codewords(h5fname):
    h5 = h5py.File(h5fname,'r')
    for nm, sub in zip(['yag','vcc'],[1,2]):
        plt.subplot(1,2,sub)
        box_idx = h5['label_%s'%nm][:]==1
        nobox_idx = h5['label_%s'%nm
        ][:]==0
        codeword2 = h5['%s_codeword2'%nm][:]
        codeword2_box = codeword2[box_idx,:]
        codeword2_nobox = codeword2[nobox_idx,:]
        toplot = np.zeros(codeword2.shape, dtype=np.float32)
        num_with_box = np.sum(box_idx)
        toplot[0:num_with_box,:] = codeword2_box
        toplot[num_with_box:,:] = codeword2_nobox
        ASPECT=30.0
        plt.imshow(toplot, interpolation='none', aspect=ASPECT)
        plt.title('%s - the %d codwords with box at top, %d without at bottom, aspect=%.1f' % 
                  (nm, num_with_box, np.sum(nobox_idx), ASPECT))
    plt.pause(.1)
    raw_input('hit enter')

if __name__ == '__main__':
    if len(sys.argv)==2:
        h5fname = sys.argv[1]
    else:
        print("setting h5fname to accbeam_codewords.h5, give command line arg to override")
        h5fname = 'acc_beam_codewords.h5'
    view_codewords(h5fname)
