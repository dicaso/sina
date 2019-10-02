# For pan-cancer research paper on importance of custom word embeddings
## Imports
## matplotlib import that needs to run before all else
import matplotlib
if __name__ == '__main__': # in interactive case
    matplotlib.rcParams['figure.max_open_warning'] = 50
else: # for headless subprocesses in multiprocessing
    matplotlib.use('pdf')
    
## general
from sina.documents import PubmedCollection, PubmedQueryResult
import sina.config
from collections import OrderedDict
import numpy as np, pandas as pd
import logging, os, time, zipfile, argparse, shutil, shelve
import gensim
import matplotlib.pyplot as plt
import dill as pickle

def corpus_workflow(corpus,settings,ext_embeddings):
    cancertype, corpus = corpus
    logging.info(cancertype)
    corpus.transform_text(preprocess=True,method='tfid')
    corpus.analyze_mesh(topfreqs=settings.topmesh)
    corpus.gensim_w2v(vecsize=settings.w2vecsize)
    #corpus.k_means_embedding(k=settings.k_clusters)
    corpus.predict_meshterms(model='svm', kmeans_only_freqs=False, rebalance='oversample')
    # Transform X for embedded processing
    corpus.transform_text(method='idx')
    corpus.nn_keras_predictor(model='cnn',embedding_trainable=False)
    print(corpus.meshtop,corpus.meshtop_nn,sep='\n')
    corpus.nn_grid_search(ext_embeddings, n_jobs=1)
    # embedding by adding an external vector
    return (cancertype, corpus)

if __name__ == '__main__':
    # Argparse
    parser = argparse.ArgumentParser()
    ## general settings
    parser.add_argument('--test', nargs='?', const=True)
    parser.add_argument('--clearcache', nargs='?', const=True)
    parser.add_argument('--debug', nargs='?', const=True, default=False)
    parser.add_argument('--interactive', nargs='?', const=True, default=False)
    parser.add_argument('--parallel', type=int) # if ==-1 use one CPU/cancertype
    parser.add_argument('--parallel-mode', default='multiprocessing') # options multiprocessing, slurm
    parser.add_argument('--parallel-job-id', type=int)     # only if --parallel-mode slurm
    ## ML/NN settings
    parser.add_argument('--w2vecsize', default = 100, type=int)
    parser.add_argument('--k_clusters', default = 100, type=int)
    parser.add_argument('--topmesh', default = 10, type=int)
    parser.add_argument('--outputdir')
    settings = parser.parse_args()

    # Set mainprocess to True if not a child process
    mainprocess = not settings.parallel_job_id

    # matplotlib interactive
    if mainprocess and not settings.interactive:
        matplotlib.use('pdf')
    
    # Prepare outputdir
    saveloc = settings.outputdir or os.path.join(
        sina.config.appdirs.user_data_dir,
        'topicmining',
        time.strftime("%Y%m%d-%H%M%S")
    )
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)
        print('created dir', saveloc)
    def savefig(fig,filename):
        fig.savefig(os.path.join(saveloc,filename+'.png'))
        fig.savefig(os.path.join(saveloc,filename+'.pdf'))
    
    # Prepare cache dir
    cachedir = sina.config.appdirs.user_cache_dir
    if (mainprocess and settings.clearcache and os.path.exists(cachedir) and
            not settings.parallel_job_id):
        shutil.rmtree(cachedir)
    if not os.path.exists(cachedir): os.mkdir(cachedir)
    
    # Logging
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        #filename=os.path.join(saveloc,'sina.log'), filemode='w',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(saveloc,'sina.log'))
        ],
        format='%(levelname)s @ %(name)s [%(asctime)s] %(message)s'
    )
    if mainprocess: logging.info('Settings: %s', settings)
        
    ## cancertypes
    ## https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga/studied-cancers
    if mainprocess:
        cancertypes = OrderedDict((
            #(ref_name, (aliasses for pubmed query...)))
            ('Acute Myeloid Leukemia', ()),
            ('Adrenocortical Carcinoma', ()),
            ('Bladder Urothelial Carcinoma', ()),
            ('Breast Ductal Carcinoma', ()),
            ('Breast Lobular Carcinoma', ()),
            ('Cervical Carcinoma', ()),
            ('Cholangiocarcinoma', ()),
            ('Colorectal Adenocarcinoma', ()),
            ('Esophageal Carcinoma', ()),
            ('Gastric Adenocarcinoma', ()),
            ('Glioblastoma Multiforme', ()),
            ('Head and Neck Squamous Cell Carcinoma', ()),
            ('Hepatocellular Carcinoma', ()),
            ('Kidney Chromophobe Carcinoma', ()),
            ('Kidney Clear Cell Carcinoma', ()),
            ('Kidney Papillary Cell Carcinoma', ()),
            ('Lower Grade Glioma', ()),
            ('Lung Adenocarcinoma', ()),
            ('Lung Squamous Cell Carcinoma', ()),
            ('Mesothelioma', ()),
            ('Ovarian Serous Adenocarcinoma', ()),
            ('Pancreatic Ductal Adenocarcinoma', ()),
            ('Paraganglioma', ('Pheochromocytoma',)), #'Paraganglioma & Pheochromocytoma'
            ('Prostate Adenocarcinoma	', ()),
            ('Sarcoma', ()),
            ('Skin Cutaneous Melanoma', ()),
            ('Testicular Germ Cell Cancer', ()),
            ('Thymoma', ()),
            ('Thyroid Papillary Carcinoma', ()),
            ('Uterine Carcinosarcoma', ()),
            ('Uterine Corpus Endometrioid Carcinoma', ()), # no cases in pub testset
            ('Uveal Melanoma', ()),
        )) if not settings.test else OrderedDict((
            ('Acute Myeloid Leukemia', ()),
            ('Adrenocortical Carcinoma', ())
        ))
        pickle.dump(
            cancertypes,
            open(os.path.join(cachedir,'cancertypes.pckl'),'wb')
        )
    else:
        # Load cancertypes from cache
        cancertypes = pickle.load(open(os.path.join(cachedir,'cancertypes.pckl'),'rb'))
        
    # Load corpora
    if mainprocess:
        logging.info('Loading pubmed collection')
        pmc = PubmedCollection('pubmed','~/pubmed')
        corpora = {
            ct: PubmedQueryResult(
                results=pmc.query_document_index(ct),corpus=pmc,
                saveloc = os.path.join(saveloc, ct.replace(' ',''))
            ) for ct in cancertypes
        }
        corpora_shelve = shelve.open(os.path.join(cachedir, 'corpora.shlv'))
        for ct in corpora: corpora_shelve[ct] = corpora[ct]
        corpora_shelve.close()
    else: corpora = shelve.open(os.path.join(cachedir, 'corpora.shlv'))
    
    # Generic embeddings
    ## Create PubmedQueryResult that merges all specific PubmedQueryResults
    if mainprocess:
        logging.info('preparing allcancers embedding')
        allcancers_training = pd.concat(
            [corpora[ct].results for ct in cancertypes],
            sort=False
        ).reset_index()
        allcancers_training.drop_duplicates('pmid', inplace=True)
        allcancers_testing = pd.concat(
            [corpora[ct].results_test for ct in cancertypes]
        ).reset_index()
        allcancers_testing.drop_duplicates('pmid', inplace=True)
        # Remove allcancers_testing papers that are in training
        # because they were in training for another cancer type
        allcancers_testing = allcancers_testing[
            ~allcancers_testing.pmid.isin(allcancers_training.pmid)
        ].copy()
        allcancers_testing.set_index('pmid', inplace=True)
        allcancers = PubmedQueryResult(
            results = allcancers_training,
            test_fraction = allcancers_testing,
            corpus = pmc,
            saveloc = os.path.join(saveloc, 'all_combined')
        )
        del allcancers_training, allcancers_testing
        # Process allcancers in the same way as subcorpora
        allcancers.transform_text(preprocess=True)
        allcancers.gensim_w2v(vecsize=settings.w2vecsize)
        #allcancers.k_means_embedding(k=settings.k_clusters)
        allcancers.analyze_mesh(topfreqs=settings.topmesh)
        #allcancers.predict_meshterms(kmeans_only_freqs=False)
        #allcancers.nn_keras_predictor(model='cnn',embedding_trainable=False)
        pickle.dump(
            allcancers,
            open(os.path.join(cachedir,'allcancers.pckl'),'wb')
        )
    else:
        allcancers = pickle.load(open(os.path.join(cachedir,'allcancers.pckl'),'rb'))
    
    ## GloVe
    logging.info('preparing glove embedding')
    ### download wiki trained glove
    glovepth = os.path.join(
        sina.config.appdirs.user_data_dir,
        'glove.6B.zip'
    )
    if not os.path.exists(glovepth[:-3]+'100d.w2v.txt'):
        import urllib.request
        gloveurl = "http://nlp.stanford.edu/data/glove.6B.zip"
        urllib.request.urlretrieve(gloveurl, glovepth)
        # Convert to gensim word2vec format
        with zipfile.ZipFile(glovepth) as zglove:
            zglove.extract('glove.6B.100d.txt', os.path.dirname(glovepth))
        from gensim.scripts import glove2word2vec
        glove2word2vec.glove2word2vec(glovepth[:-3]+'100d.txt',glovepth[:-3]+'100d.w2v.txt')
    glovemdl = gensim.models.KeyedVectors.load_word2vec_format(glovepth[:-3]+'100d.w2v.txt')

    # Multiprocessing logic
    if mainprocess and settings.parallel and settings.parallel_mode == 'multiprocessing':
        import multiprocessing as mp
        from functools import partial
        logging.info('starting pool of %s workers', settings.parallel)
        with mp.Pool(
                len(cancertypes) if settings.parallel == -1 else settings.parallel
                ) as pool:
            corpora = dict(pool.map(
                partial(
                    corpus_workflow, settings=settings,
                    ext_embeddings=[allcancers.embedding, glovemdl]),
                corpora.items()
            ))
    else: #execute everythin with one CPU
        for ct in cancertypes:
            corpus_workflow(
                corpus=(ct,corpora[ct]),settings=settings,
                ext_embeddings=[allcancers.embedding, glovemdl]
            )

    if mainprocess:
        docoverlap = np.zeros((len(cancertypes),len(cancertypes)))
        for i,cti in enumerate(cancertypes):
            for j,ctj in enumerate(cancertypes):
                docoverlap[i,j] = len(corpora[cti].results.index.intersection(
                  corpora[ctj].results.index))/len(corpora[cti].results.index)
        fig, ax = plt.subplots()
        plt.imshow(docoverlap, cmap='viridis')
        plt.colorbar()
        savefig(fig,'corpora_overlap')
        
        customs = []
        generics = []
        gloves = []
        for ct in cancertypes:
            corpus = corpora[ct]
            print(
                ct,
                len(corpus.results),
                'generic embedding' if corpus.nn_grid_result.best_params_['embedding'] is allcancers.embedding 
                else ('glove embedding' if corpus.nn_grid_result.best_params_['embedding'] is glovemdl else 'custom embedding')
            )
            if corpus.nn_grid_result.best_params_['embedding'] is allcancers.embedding:
                generics.append(ct)
            elif corpus.nn_grid_result.best_params_['embedding'] is glovemdl:
                gloves.append(ct)
            else:
                customs.append(ct)
        
        # Corpus size category
        fig, ax = plt.subplots()
        ax.scatter([len(corpora[ct].results) for ct in generics], [0]*len(generics), label='generic')
        ax.scatter([len(corpora[ct].results) for ct in customs], [1]*len(customs), label='custom')
        ax.scatter([len(corpora[ct].results) for ct in gloves], [2]*len(gloves), label='glove')
        ax.legend()
        ax.set_title('Corpus size and the relevance of a custom embedding')
        ax.set_xlabel('Corpus size (#)')
        savefig(fig,'corpus_size')
        ## optional look at embedding_trainable
        
        from scipy import stats
        cscores = [corpora[ct].nn_grid_result.best_score_ for ct in customs]
        gscores = [corpora[ct].nn_grid_result.best_score_ for ct in generics]
        glscores = [corpora[ct].nn_grid_result.best_score_ for ct in gloves]
        logging.info(
            'generic mean+var %s (%s)\ncustom mean+var %s (%s)\nglove mean+var %s (%s)\nttest %s',
            np.mean(gscores), np.var(gscores),
            np.mean(cscores), np.var(cscores),
            np.mean(glscores), np.var(glscores),
            stats.f_oneway(cscores, gscores, glscores) #stats.ttest_ind(cscores, gscores)    
        )

        # Score vs size
        fig, ax = plt.subplots()
        ax.scatter(gscores,[len(corpora[ct].results) for ct in generics],label='generic')
        ax.scatter(cscores,[len(corpora[ct].results) for ct in customs],label='custom')
        ax.scatter(glscores,[len(corpora[ct].results) for ct in gloves],label='glove')
        ax.set_xlabel('NN score')
        ax.set_ylabel('Corpus size (#)')
        savefig(fig,'score_vs_corpus_size')
        
        # Maximum overlap with other corpus
        fig, ax = plt.subplots()
        ax.scatter(
            [len(corpora[ct].results) for ct in generics],
            np.where(docoverlap==1, 0, docoverlap).max(axis=1)[[ct in generics for ct in cancertypes]], label='generic')
        ax.scatter(
            [len(corpora[ct].results) for ct in customs],
            np.where(docoverlap==1, 0, docoverlap).max(axis=1)[[ct in customs for ct in cancertypes]], label='custom')
        ax.legend()
        ax.set_title('Corpus size and maximum overlap with other corpora')
        ax.set_xlabel('Corpus size (#)')
        ax.set_ylabel('Maximum overlap with other corpus (%)')
        savefig(fig,'corpus_size_overlap')
        
        #3d
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter([len(corpora[ct].results) for ct in generics], np.where(docoverlap==1, 0, docoverlap).max(axis=1)[[ct in generics for ct in cancertypes]], gscores, marker='o', label='generic')
        # ax.scatter([len(corpora[ct].results) for ct in customs], np.where(docoverlap==1, 0, docoverlap).max(axis=1)[[ct in customs for ct in cancertypes]], cscores, marker='^', label='custom')
        # ax.set_xlabel('Corpus size (#)')
        # ax.set_ylabel('Maximum overlap with other corpus (%)')
        # ax.set_zlabel('NN score')
        
        #Maximum overlap with other corpus (%) - marker sized corpus
        from scipy.interpolate import interp1d
        sizemap = interp1d([min([len(corpora[ct].results) for ct in cancertypes]), max([len(corpora[ct].results) for ct in cancertypes])],[20,100])
        ax.scatter(
            np.where(docoverlap==1, 0, docoverlap).max(axis=1)[[ct in generics for ct in cancertypes]],
            gscores, s=[sizemap(len(corpora[ct].results)) for ct in generics], marker='o', label='generic')
        ax.scatter(
            np.where(docoverlap==1, 0, docoverlap).max(axis=1)[[ct in customs for ct in cancertypes]],
            cscores, s=[sizemap(len(corpora[ct].results)) for ct in customs], marker='^', label='custom')
        ax.legend()
        ax.set_title('Overlap and score')
        ax.set_ylabel('NN score')
        ax.set_xlabel('Maximum overlap with other corpus (%)')
        savefig(fig,'overlap_score')
        
    
