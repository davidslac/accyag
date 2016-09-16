from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import h5py

import matplotlib.pyplot as plt
plt.ion()
plt.show()

sys.path.append('/reg/neh/home/davidsch/github/davidslac/psana-mlearn')
import  psmlearn
import psmlearn.boxutil as boxutil

def run_through_all(nm, imgs, predict, truth):
    for img, predictbx, truthbx in zip(imgs, predict, truth):
        plt.clf()
        plt.imshow(img, interpolation='none')
        print("predictbx=%s" % boxutil.box_to_str(predictbx))
        boxutil.plot_box(predictbx, plt, 'w')
        boxutil.plot_box(truthbx, plt, 'r')
        plt.ylim([0, img.shape[0]])
        plt.xlim([0, img.shape[1]])
        plt.title("nm=%s inter/union=%.2f" % (nm, boxutil.intersection_over_union(predictbx, truthbx)))
        plt.pause(.1)
        raw_input('hit enter')


def view(regress_results, acc_codewords, view=False, viewall=False):
    h5regress=h5py.File(regress_results,'r')
    h5cw=h5py.File(acc_codewords,'r')
    for nm in ['yag','vcc']:
        predict=h5regress['%s_predict'%nm][:]
        include=h5cw['label_%s'%nm][:]
        incIdx=1==include
        truth=h5cw['%sbox'%nm][:]
        imgs=h5cw['%s_preprocessed_imgs'%nm][:]

        predict = predict[incIdx]
        truth = truth[incIdx]
        imgs = imgs[incIdx]

        msg = 'nm=%s %s inter/union accuracies: ' % (nm, regress_results)
        scores = [boxutil.intersection_over_union(predictbx, truthbx) for predictbx, truthbx in zip(predict,truth)]
        assert len(scores)==imgs.shape[0]
        accs=[]
        threshs=[.5,.2,.01]
        for th in threshs:
            hits = np.array([sc>th for sc in scores])
            acc = np.sum(hits)/float(len(hits))
            msg += ' th=%.2f acc=%.2f' % (th, acc)
            accs.append(acc)
        print(msg)
        if viewall:
            run_through_all(nm, imgs, predict, truth)

        if view:
            fig = plt.figure(20, figsize=(18,12))
            rois_bb = [boxutil.containing_box(truthbx, predictbx) for truthbx, predictbx in zip(truth, predict)]
            rois = [boxutil.extract_box(img,bx) for img,bx in zip(imgs, rois_bb)]
            idx2bbox={}
            grid = psmlearn.make_grid(rois, idx2bbox=idx2bbox)
            plt.clf()
            plt.imshow(grid, interpolation='none')
            for idx, predictbx, truthbx, bb in zip(range(len(rois)), predict, truth, rois_bb):
                grid_y, _, grid_x, _ = idx2bbox[idx][:]
                img_y, _, img_x, _ = bb[:]
                dy = grid_y-img_y
                dx = grid_x-img_x
                truthbx_translated = boxutil.translate(truthbx, y=dy, x=dx)
                predictbx_translated = boxutil.translate(predictbx, y=dy, x=dx)
                boxutil.plot_box(truthbx_translated, plt, colorstr='w')
                boxutil.plot_box_clipped_to(predictbx_translated, plt, clipbx=idx2bbox[idx], colorstr='r')
            plt.xlim([0,grid.shape[1]])
            plt.ylim([0,grid.shape[0]])
            plt.title(msg)
            plt.pause(.1)
            raw_input('hit enter')
