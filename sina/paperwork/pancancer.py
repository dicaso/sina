# For pan-cancer research paper on importance of custom word embeddings
## Imports
from sina.documents import PubmedCollection, PubmedQueryResult
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

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

for ct in cancertypes:
    print(ct)
    corpora[ct]
    corpora[ct].gensim_w2v(vecsize=100,doclevel=True)

#qr.k_means_embedding(k=100)
#qr.analyze_mesh(topfreqs=10,getmeshnames=True)
#qr.predict_meshterms(kmeans_only_freqs=False)
