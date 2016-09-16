import os
import sys
import h5py
import argparse

sys.path.append('/reg/neh/home/davidsch/github/davidslac/psana-mlearn')
import psmlearn

import preprocess
import beamdata
import regressresults

#MLEARNDIR = '/reg/d/ana01/temp/davidsch/mlearn'
#PROJECT_DIR = os.path.join(MLEARNDIR, 'acc_beam_locate')
#VGG16_DIR = os.path.join(MLEARNDIR, 'vgg16')

programDescription = '''
pipeline for localization analaysis of acc beam yag vcc screens
'''

programDescriptionEpilog = '''
Steps are
1. create codewords, use --skipcw if already created
  a. initialize beamdata to read filenums, use --f to limit
  b. use -n for debugging, just do a few samples 
     (make sure enough to get a box) 
  b. use --view to see summary plots, --viewall to see shot per shot plots
  c. use --save to save plots, with specified number for shot by shot. 
     When save is specified, plots are not interactive.
  d. use --subbkg for file 4
  e. specify --alg for the preprocessing algorithm for preparing imgs for vgg16
2. view codewords TODO (maybe tsne also?)
3. regress, use --skipregress to skip this
  a. use --fsvar for variance based feature selection
4. view results, give --results for text, and --view, --viewall for plots
'''

parser = argparse.ArgumentParser(description=programDescription,
                                 epilog=programDescriptionEpilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-f', '--filenums', type=str, help='comma sep list of files to do, default is 1,2,4', default='1,2,4')
parser.add_argument('-n','--nn', type=int, help='debugging, only do the first nn samples', default=0)
parser.add_argument('--alg', type=str, help='algorithm to do. One of %r' % preprocess.ALGS.keys(), default=None)
parser.add_argument('--prefix', type=str, help='prefix for filenames', default='')
parser.add_argument('--view', action='store_true', help='view results')
parser.add_argument('--viewall', action='store_true', help='view all results (include per shot plots)')
parser.add_argument('--vgg16weights', type=str, help='weights file for vgg16', default='vgg16_weights.npz')
parser.add_argument('--save', type=int, help='number of plots to save for per image plots, save all one-shot plots', default=0)
parser.add_argument('--force', action='store_true', help='overwrite existing filenames')
parser.add_argument('--dbgcodewords', action='store_true', help='debug generation of codewords')
parser.add_argument('--subbkg', action='store_true', help='subtract background for yag/vcc if provided')
parser.add_argument('--fsvar', type=float, help='minimum variance for feature selection with regression. Default is None', default=-1)
parser.add_argument('--skipcw', action='store_true', help='skip generation of codewords, use existing codewords file')
parser.add_argument('--skipregress', action='store_true', help='skip regression, use existing regress file')
parser.add_argument('--results', action='store_true', help='view regression results, use existing regress file')

args = parser.parse_args()
assert args.prefix, "provide prefix"
filenums = map(int,args.filenums.split(','))

codewords_fname = '%s-%s' % (args.prefix, 'acc_beam_codewords.h5')
if args.skipcw:
    codeword_datasets = {'yag':('yag_codeword1','yag_codeword2'),
                         'vcc':('vcc_codeword1','vcc_codeword2')}
else:
    assert args.alg and args.alg in preprocess.ALGS.keys(), "provide algorithm, one of %r, see preprocess.py" % preprocess.ALGS.keys()
    preprocess = preprocess.PreProcess(alg=args.alg, 
                                       final_shape=(224, 224))

    accdata = beamdata.BeamData(preprocess=preprocess, 
                                datadir='data', 
                                subbkg=args.subbkg, 
                                filenums=filenums,
                                prefix=args.prefix,
                                force=args.force,
                                nn=args.nn)

    accdata.loadall(reload=False)

    if args.view or args.save>0:
        accdata.plotROIs(save=args.save, view=args.view)
        accdata.plots(do_sigprocess=True, save=args.save, view=args.view)
        accdata.max_stats()

    codeword_datasets = psmlearn.vgg16.write_codewords(dataloader=accdata,
                                                       output_fname=codewords_fname,
                                                       weights= args.vgg16weights,
                                                       dbg=args.dbgcodewords,
                                                       force=args.force)

assert codeword_datasets['yag']==('yag_codeword1','yag_codeword2'), "vgg16.write_codewords did not return expected datasets for yag, returned %s" % str(codeword_datasets['yag'])
assert codeword_datasets['vcc']==('vcc_codeword1','vcc_codeword2'), "vgg16.write_codewords did not return expected datasets for vcc, returned %s" % str(codeword_datasets['yag'])
assert os.path.exists(codewords_fname), "output file: %s not written by vgg16.write_codewords" % codewords_fname 

h5 = h5py.File(codewords_fname,'r')
for dsname in ['yagbox', 'vccbox', 'label_yag', 'label_vcc']:
    assert dsname in h5.keys(), "expected dataset %s not in %s" % (dsname, codewords_fname)
h5.close()

problems = {'yag':{'X_ds':('yag_codeword1','yag_codeword2'),
                   'Y_ds':('yagbox',),
                   'include_ds':'label_yag'},
            'vcc':{'X_ds':('vcc_codeword1','vcc_codeword2'),
                   'Y_ds':('vccbox',),
                   'include_ds':'label_vcc'}
        }

regress_fname = '%s-regress.h5' % (args.prefix,)
if args.skipregress:
    pass
else:
    psmlearn.regress.regress(inputh5=codewords_fname,
                             outputh5=regress_fname,
                             problems=problems,
                             variance_feature_select=args.fsvar,
                             force=args.force)

if args.results:
    regressresults.view(regress_fname, codewords_fname, view=args.view, viewall=args.viewall)
                 
