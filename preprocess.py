import numpy as np
from scipy.signal import medfilt2d, convolve2d
from scipy.ndimage.filters import maximum_filter
from scipy.misc import imresize

_KERN_ONE_3x3 = np.ones((3,3), dtype=np.float32)
_DEFAULT_FINAL_SHAPE=(224,224)

#################### helpers

def n_times_bigger(img, factor, target_shape):
    '''return True if img is factor bigger in both x/y, i.e, 
    img.shape=(10,10) and target_shape=(3,3) then
    return True for factor=1,2,3 but not 4,5,...
    '''
    return img.shape[0] >= factor*target_shape[0] and img.shape[1] >= factor*target_shape[1]

def max_reduce(img, target_shape):
    '''reduce image to something bigger or equal to target_shape by taking the
    max over each block. I'e, if img.shape=(10,10) and target_shape=(3,3), 
    result is A with shape=(3,3) where
    A[0,0]=max(img[0:3,0:3]
    A[1,1]=max(img[0:3,3:6]
    A[1,2]=max(img[0:3,6:9]
    A[1,0]=max(img[3:6,0:3]
    ...
    '''
    factor = 0
    while n_times_bigger(img, factor+1, target_shape=_DEFAULT_FINAL_SHAPE):
        factor += 1

    block = None
    for ii in range(factor):
        for jj in range(factor):
            sl = img[ii::factor,jj::factor]
            if block is None:
                block = np.zeros((factor*factor, sl.shape[0], sl.shape[1]), dtype=np.float32)
            block[factor*ii+jj,0:sl.shape[0],0:sl.shape[1]]=sl[:,:]
    return np.max(block, axis=0)

def threshold(img, lower, upper):
    return np.minimum(upper,np.maximum(lower,img))

######### PREPROCESSING ALGORITHMS ########## 

def preProcessImgNone(nm, img, final_shape=_DEFAULT_FINAL_SHAPE):
    return imresize(img, final_shape, interp='lanczos', mode='F')

def preProcessImgDeNoiseLog(nm, img, final_shape=_DEFAULT_FINAL_SHAPE):
    '''a little de-noising, log transform, reduce, then scale up 
    '''
    assert nm in ['vcc','yag']
    img = img.astype(np.float32)
    img = medfilt2d(img,3)
    img = np.log(1+np.maximum(0.0,img))
    img = imresize(img, final_shape, interp='lanczos', mode='F')
    mxImg = np.max(img)
    mapTo255 = max(np.log(2000.0), mxImg)
    img *= 255.0/mapTo255
    return img 

def preProcessImgDenoiseGainMaxReduceLog(nm, img, final_shape=_DEFAULT_FINAL_SHAPE):
    '''meant for yag's with faint signal, 
    boost up signal of beam by summing box of pixels
    denoise by median filter
    apply to vcc as well
    '''
    assert nm in ['vcc','yag']
    img = img.astype(np.float32)
    img = medfilt2d(img,3)
    img = convolve2d(img, _KERN_ONE_3x3, mode='same')
    img = medfilt2d(img, 3)
    img = max_reduce(img, target_shape=final_shape)
    img = imresize(img, final_shape, interp='lanczos', mode='F')
    img = np.log(1+np.maximum(0.0,img))
    mxImg = np.max(img)
    mapTo255 = max(np.log(2000.0), mxImg)
    img *= 255.0/mapTo255
    return img 

    
################ Main code #####################
ALGS = {'none':preProcessImgNone,
        'denoise-log':preProcessImgDeNoiseLog,
        'denoise-max-log':preProcessImgDenoiseGainMaxReduceLog}

####################
def _preProcessAll(nm, orig_imgs, alg, final_shape=_DEFAULT_FINAL_SHAPE):
    ##### helpers

    assert alg in ALGS.keys(), "alg must be one of %s" % ALGS.keys()
    preprocessImageFn = ALGS[alg]
    assert nm in ['yag','vcc'], "nm must be yag or vcc"
    imgs = np.zeros((len(orig_imgs), final_shape[0], final_shape[1]), dtype=np.float32)
    for idx in range(imgs.shape[0]):
        imgs[idx,:,:] = preprocessImageFn(nm, orig_imgs[idx], final_shape=final_shape)
    grand_mean = np.mean(imgs)
    imgs -= grand_mean
    return imgs

class PreProcess(object):
    def __init__(self, alg, final_shape=_DEFAULT_FINAL_SHAPE):
        assert alg in ALGS, "alg must be one of %r" % ALGS.keys()
        self.alg=alg
        self.final_shape= final_shape

    def preProcessAll(self, nm, imgs, alg=None):
        if alg is None:
            alg = self.alg
        return _preProcessAll(nm, imgs, alg, self.final_shape)
