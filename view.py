from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
plt.ion()
plt.show()

def plot_box(bx):
    ymin,ymax,xmin,xmax=bx[:]
    plt.plot([xmin,xmin,xmax,xmax,xmin],[ymin,ymax,ymax,ymin,ymin], 'w')

class Data(object):
    def __init__(self):
#        self.h5filename = '/reg/d/ana01/temp/davidsch/mlearn/accbeam.h5' 
        self.h5filename = 'accbeam.h5'
    def load_from_mat(self):
        data = {'yag':{'img':[],
                       'box':[],
                       'label':[],
                       'file':[]},
                'vcc':{'img':[],
                       'box':[],
                       'label':[],
                       'file':[]}
                }
        for dig in [1,2]:
            mat = sio.loadmat('labeledimg%d.mat' % dig)
            keys = [ky for ky in mat.keys() if not ky.startswith('__')]
            assert set(keys)==set(['yagImg', 'yagbox', 'vccImg', 'vccbox'])
            yagImg = mat['yagImg'][0,:]
            vccImg = mat['vccImg'][0,:]
            yagbox = mat['yagbox'][0,:]
            vccbox = mat['vccbox'][0,:]
            assert len(yagImg)==len(yagbox)
            assert len(vccImg)==len(vccbox)
            print("file %d has %d yag, and %d vcc" % (dig, len(yagImg), len(vccImg)))
            for yag, vcc, yagbx, vccbx in zip(yagImg, vccImg, yagbox, vccbox):
                data['yag']['img'].append(yag)
                data['vcc']['img'].append(vcc)
                data['yag']['box'].append(yagbx)
                data['vcc']['box'].append(vccbx)
                data['yag']['file'].append(dig)
                data['vcc']['file'].append(dig)
                label = 0
                if len(yagbx)>0:
                    label = 1
                    assert len(vccbx)>0
                data['yag']['label'].append(label)
                data['vcc']['label'].append(label)
                print("wrote yag/vcc/boxes")
        self.data = data

    def save_to_h5(self):
        h5=h5py.File(self.h5filename, 'w')
        num = len(self.data['yag']['img'])
        label = np.array(self.data['yag']['label']).astype(np.int32)
        filenum = np.array(self.data['yag']['file']).astype(np.int32)
        assert np.all(label==np.array(self.data['vcc']['label']).astype(np.int32))
        assert np.all(filenum==np.array(self.data['vcc']['file']).astype(np.int32))
        h5['label'] = label
        h5['file'] = filenum

        for name in ['yag','vcc']:
            img0 = self.data[name]['img'][0]
            images = np.zeros((num,img0.shape[0],img0.shape[1]), dtype=img0.dtype)
            boxes = np.zeros((num,4), dtype=np.uint16)
            for row in range(num):
                images[row,:,:] = self.data[name]['img'][row]
                if label[row]==1:
                    boxes[row,:] = self.data[name]['box'][row][0,:]
            h5[name]=images
            h5[name+'box']=boxes
        h5.close()

    def clipped_means(self):
        h5 = h5py.File(self.h5filename, 'r')
        yags = np.minimum(255.0, h5['yag'][:])
        vccs = np.minimum(255.0, h5['vcc'][:])
        print("yags, shape=%r mean=%.2f" % (yags.shape, np.mean(yags)))
        print("vccs, shape=%r mean=%.2f" % (vccs.shape, np.mean(vccs)))
# this returns
#yags, shape=(142, 1040, 1392) mean=1.26
#vccs, shape=(142, 480, 640) mean=0.36
     
    def max_stats(self):
        '''what is max value in yag/vcc inside the boxes? vs outside?
        '''
        h5 = h5py.File(self.h5filename, 'r')
        num = len(h5['label'])
        yags = h5['yag'][:]
        vccs = h5['vcc'][:]
        yagboxs = h5['yagbox'][:]
        vccboxs = h5['vccbox'][:]
        labels = h5['label'][:]

        def maxValueInBox(imgs, boxs,row):
            ymin,ymax,xmin,xmax=boxs[row,:]
            bx = imgs[row,ymin:ymax,xmin:xmax]
            return np.max(bx)

        mx_vcc_box = 0
        mx_yag_box = 0
        mx_vcc = 0
        mx_yag = 0
        for row in range(num):
            if not labels[row]: continue
            mx_vcc_box = max(mx_vcc_box, maxValueInBox(vccs, vccboxs,row))
            mx_yag_box = max(mx_yag_box, maxValueInBox(yags, vccboxs,row))
            mx_vcc = max(mx_vcc, np.max(vccs[row,:,:]))
            mx_yag = max(mx_yag, np.max(yags[row,:,:]))
        print("stats: mx_vcc_box=%.1f mx_yag_box=%.1f mx_vcc=%.1f mx_yag=%.1f" %
              (mx_vcc_box, mx_yag_box, mx_vcc, mx_yag))
# returns
# stats: mx_vcc_box=255.0 mx_yag_box=14.0 mx_vcc=255.0 mx_yag=1114.0

    def plot_from_h5(self):
        h5 = h5py.File(self.h5filename, 'r')
        num = len(h5['label'])
        plt.figure(1, figsize=(18, 12))
        yags = h5['yag'][:]
        vccs = h5['vcc'][:]
        labels = h5['label'][:]
        fileNums = h5['file'][:]
        yagboxs = h5['yagbox'][:]
        vccboxs = h5['vccbox'][:]
        for row in range(num):
            yag = yags[row,:,:]
            vcc = vccs[row,:,:]
            label = labels[row]
            fileNum = fileNums[row]
            yagbox = yagboxs[row,:]
            vccbox = vccboxs[row,:]

            plt.clf()
            plt.subplot(1,2,1)
            plt.imshow(yag)
            if label:
                plot_box(yagbox)
            plt.xlim([0,yag.shape[1]])
            plt.ylim([0,yag.shape[0]])
            plt.title('yag')

            plt.subplot(1,2,2)
            plt.imshow(vcc)
            if label:
                plot_box(vccbox)
            plt.xlim([0,vcc.shape[1]])
            plt.ylim([0,vcc.shape[0]])
            plt.title('vcc')
            plt.pause(.1)
            raw_input('file=%d img=%d yag.shape=%r vcc.shape=%r yag min/max = [%.1f,%.1f] vcc min/max = [%.1f %.1f] hit enter' %
                      (fileNum, row, yag.shape, vcc.shape, np.min(yag), np.max(yag), np.min(vcc), np.max(vcc)))

if __name__ == '__main__':
    accdata = Data()
    accdata.load_from_mat()
    accdata.save_to_h5()
    accdata.plot_from_h5()
    accdata.max_stats()
    accdata.clipped_means()
