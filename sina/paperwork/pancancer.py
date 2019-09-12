# For pan-cancer research paper on importance of custom word embeddings
## Imports
from sina.documents import PubmedCollection, PubmedQueryResult
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# ML/NN settings
w2vecsize  = 100
k_clusters = 100
topmesh    = 10

## cancertypes
## https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga/studied-cancers
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
    #('Uterine Corpus Endometrioid Carcinoma', ()), # no cases in pub testset
    ('Uveal Melanoma', ()),
))

# Load corpora
pmc = PubmedCollection('pubmed','~/pubmed')
corpora = {
    ct: PubmedQueryResult(results=pmc.query_document_index(ct),corpus=pmc)
    for ct in cancertypes
}

# Create PubmedQueryResult that merges all specific PubmedQueryResults
allcancers_training = pd.concat(
        [corpora[ct].results for ct in cancertypes]
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
allcancers = PubmedQueryResult(
    results = allcancers_training,
    test_fraction = allcancers_testing,
    corpus = pmc
)
del allcancers_training, allcancers_testing
allcancers.gensim_w2v(vecsize=w2vecsize,doclevel=True)
allcancers.k_means_embedding(k=k_clusters)
allcancers.analyze_mesh(topfreqs=topmesh,getmeshnames=True)
allcancers.predict_meshterms(kmeans_only_freqs=False)
allcancers.nn_keras_predictor(textprep=False)

for ct in cancertypes:
    print(ct)
    corpora[ct].gensim_w2v(vecsize=w2vecsize,doclevel=True)
    corpora[ct].k_means_embedding(k=k_clusters)
    corpora[ct].analyze_mesh(topfreqs=topmesh,getmeshnames=True)
    corpora[ct].predict_meshterms(kmeans_only_freqs=False)
    corpora[ct].nn_keras_predictor(textprep=False)
    
docoverlap = np.zeros((len(cancertypes),len(cancertypes)))
for i,cti in enumerate(cancertypes):
    for j,ctj in enumerate(cancertypes):
        docoverlap[i,j] = len(corpora[cti].results.index.intersection(
          corpora[ctj].results.index))/len(corpora[cti].results.index)
fig, ax = plt.subplots()
plt.imshow(docoverlap)
