# For neuroblastoma research paper on clustering high-risk MYCN amplified
# versus non-amplified related statements
# workstation command that generated results:
# python3 -m sina.paperwork.neuroblastoma


def main():
    from sina.documents import PubmedCollection, PubmedQueryResult
    from argetype import ConfigBase
    from bidali.LSD.dealer.genenames import get_genenames
    from bidali.LSD.dealer.celllines import get_NB39
    import networkx as nx
    from node2vec import Node2Vec
    import pandas as pd
    from scipy.stats import spearmanr

    # Exposing variables that are created in the script for debugging
    # and further development
    global settings, nbresults, mycn_grams, mycn_gram_simils, genecorrelations, model, topgc

    # Settings

    class Settings(ConfigBase):
        cancercorpus: bool = False  # Include cancer corpus analysis and comparison
        debug: bool = False
        mincorr: float = .7  # Minimal correlation level for building correlation network
        vecsize: int = 100  # size of the embedding vectors
        adjpval: bool = True  # adjust P values with R function
        tm_exp_corr: bool = True  # If yes, do not predict topics and use all abstracts
        interactive: bool = False  # Interactive for matplotlib

    settings = Settings()
    if settings.adjpval:
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        stats = importr('stats')

    # from sina.ranking.ligsea import Ligsea
    import matplotlib.pyplot as plt
    if settings.interactive:
        plt.ion()
    else:
        import matplotlib
        matplotlib.use('pdf')

    # retrieve documents
    pmc = PubmedCollection()
    nbcorpus = pmc.query_document_index('neuroblastoma')
    nbresults = PubmedQueryResult(
        results=nbcorpus, corpus=pmc,
        test_fraction=0 if settings.tm_exp_corr else .25
    )
    nbresults.transform_text(preprocess=True, method='tfid')
    nbresults.gensim_w2v(vecsize=settings.vecsize)
    # Retrieve 'mycn' grams
    mycn_grams = [
        w for w in nbresults.embedding.wv.vocab
        if 'mycn' in w.lower()
    ]
    mycn_gram_simils = {
        w[0] for mg in mycn_grams for w in nbresults.embedding.wv.similar_by_word(mg, 3)
    }
    print(mycn_grams)
    nbresults.vizualize_embedding(list(set(mycn_grams) | mycn_gram_simils))

    # See how many genes are represented
    genenames = get_genenames()
    genevectoravail = genenames.symbol.str.lower().isin(nbresults.embedding.wv.vocab)
    genesymbols = genenames.symbol[genevectoravail]

    if settings.tm_exp_corr:
        # Make experimental correlation network embedding
        nb39 = get_NB39()
        nb39.exprdata = nb39.exprdata[nb39.exprdata.index.isin(genesymbols)].copy()
        nb39nx = nb39.exprdata.T.corr() >= settings.mincorr  # TODO negatice correlations with abs
        # Transform to 3 columns format
        nb39nx.columns.name = 'node2'
        edges = nb39nx.stack()
        edges = edges[edges].reset_index()
        edges.columns = ['node1', 'node2', 'value']
        edges = edges[edges.node1 != edges.node2].copy()
        # Build graph
        G = nx.from_pandas_edgelist(edges, 'node1', 'node2')
        # nx.draw(G, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=15)
        node2vec = Node2Vec(
            G, dimensions=settings.vecsize, walk_length=30, num_walks=200, workers=4
        )
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Investigating embedding correlations
        genecorrelations = {}
        for gene in genesymbols:
            if gene not in model.wv.vocab or gene.lower() not in nbresults.embedding.wv.vocab:
                continue
            genecorr = pd.DataFrame(
                model.wv.similar_by_word(
                    gene,
                    len(model.wv.vocab)
                )
            ).set_index(0)
            genelit = pd.DataFrame(
                nbresults.embedding.wv.similar_by_word(
                    gene.lower(),
                    len(nbresults.embedding.wv.vocab)
                )
            ).set_index(0)
            genelit.index = genelit.index.str.upper()
            genelit = genelit[genelit.index.isin(genecorr.index)].sort_index()
            genecorr = genecorr[genecorr.index.isin(genelit.index)].sort_index()
            genecorrelations[gene] = spearmanr(genelit, genecorr)
            print(gene, genecorrelations[gene])
        genecorrelations = pd.DataFrame(genecorrelations).T
        genecorrelations.columns = ['correlation', 'pvalue']
        print(
            genecorrelations.sort_values('pvalue').head(10),
            genecorrelations.sort_values('pvalue').tail(10)
        )
        topgc = genecorrelations[genecorrelations.pvalue < .05].sort_values('pvalue')
        if settings.adjpval:
            genecorrelations.sort_values('pvalue', inplace=True)
            genecorrelations['padj'] = stats.p_adjust(genecorrelations.pvalue, method='fdr')
            # method options = "holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr"

    # Bigger embedding for comparison and to do projection of low frequency terms
    if settings.cancercorpus:
        cancercorpus = pmc.query_document_index('cancer')
        cancerresults = PubmedQueryResult(
            results=cancercorpus, corpus=pmc,
            test_fraction=0 if settings.tm_exp_corr else .25
        )
        cancerresults.transform_text(preprocess=True, method='tfid')
        cancerresults.gensim_w2v(vecsize=100)

    if not settings.tm_exp_corr:
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

        # no return statement -> returning through declaring globals


if __name__ == '__main__':
    main()
