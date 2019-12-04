# For neuroblastoma research paper on clustering high-risk MYCN amplified
# versus non-amplified related statements
# workstation command that generated results:
# python3 -m sina.paperwork.neuroblastoma


def main():
    from sina.documents import PubmedCollection, PubmedQueryResult
    # from sina.ranking.ligsea import Ligsea
    import matplotlib.pyplot as plt
    plt.ion()
    pmc = PubmedCollection()
    nbcorpus = pmc.query_document_index('neuroblastoma')
    nbresults = PubmedQueryResult(results=nbcorpus, corpus=pmc)
    nbresults.transform_text(preprocess=True, method='tfid')
    nbresults.gensim_w2v(vecsize=100)
    # Retrieve 'mycn' grams
    mycn_grams = [
        w for w in nbresults.embedding.wv.vocab
        if 'mycn' in w.lower()
    ]
    print(mycn_grams)
    # Bigger embedding to do projection of low frequency terms
    # qr_big = PubmedQueryResult(results=pmc.query_document_index('cancer'),corpus=pmc)
    # qr_big.transform_text(preprocess=True,method='tfid')
    # qr_big.gensim_w2v(vecsize=100)
    # nbresults.k_means_embedding(k=100)
    nbresults.analyze_mesh(topfreqs=10, getmeshnames=True)
    nbresults.predict_meshterms(model='svm', kmeans_only_freqs=False, rebalance='oversample')
    nbresults.nn_keras_predictor()
    nbresults.transform_text(method='idx')
    nbresults.nn_keras_predictor(model='cnn', embedding_trainable=False)
    # nbresults.nn_grid_search(qr_big.embedding)
    # lg = Ligsea('neuroblastoma', 'metasta', '/tmp/mock.csv', 'gene_col')
    # lg.retrieve_documents()
    # lg.determine_gene_associations()
    # lg.evaluate_gene_associations()
    # lg.predict_number_of_relevant_genes(surges=1)
    # lg.plot_ranked_gene_associations()
    # lg.calculate_enrichment()
    # lg.retrieve_datasets()


if __name__ == '__main__':
    main()