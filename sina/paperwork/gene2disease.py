# In preparation for Riyadt AI summit
# module to make gene-disease map
# Example: results, ds_gene_litdistance = disease_map('"cleft lip" OR "cleft palate"', 'cleft')
# TODO add adjustText to setup.py


def disease_map(disease_query, disease_substr=None, figsize=(12, 8),
                gap=1, sigcorr=.2, fontsize=16, adjustlabels=False, corpus=None):
    """Make a text mining disease map with
    in the centre a disease relevant synonym.

    In the map with polar coordinates, genes that have hight text similarity
    with the disease are shown closer to the centre. Positive correlated genes
    in green, negative in red.

    Args:
        disease_query (str): elastict-search-like query string.
        disease_substr (str): words containing this string, will be considered
          disease words and the most frequent one used to calculate correlations
          with genes.
        gap (float): In the polar plot the most dissimal genes would end up next
          to each other, with gap the distance is defined between them.
        sigcorr (float): genes with a higher absolute correlation with the disease
          word will be annotated in the plot.
        fontsize (int): Size for annotations.
        adjustLabels (bool): If True, use adjust_text, but be aware that you cannot
          readjust size after applying it, so you need to supply final figsize.
        corpus (PubmedQueryResult): Corpus with precalculated embedding.
    """
    from sina.documents import PubmedCollection, PubmedQueryResult
    from bidali.LSD.dealer.genenames import get_genenames
    from scipy.cluster.hierarchy import dendrogram, linkage
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from adjustText import adjust_text

    if not corpus:
        pmc = PubmedCollection()
        corpus = pmc.query_document_index(disease_query)
        results = PubmedQueryResult(results=corpus, corpus=pmc, test_fraction=0)
        results.transform_text(preprocess=True, method='tfid')
        results.gensim_w2v()
    else:
        results = corpus

    genenames = get_genenames()
    genevectoravail = genenames.symbol.str.lower().isin(results.embedding.wv.vocab)
    genesymbols = genenames.symbol[genevectoravail]
    print('Genes with embedding:', len(genesymbols))
    print('Vocab size', len(results.embedding.wv.vocab))
    ds_words = [
        w for w in results.embedding.wv.vocab
        if (disease_substr in w if disease_substr else disease_query in w)
    ]
    print('Disease related words', list(zip(range(len(ds_words)), ds_words)))
    # chosen_ds_word_ix = int(input('Choose disease word index: '))
    ds_gene_litdistance = pd.DataFrame(
        {
            'gene': genesymbols,
            'distance': genesymbols.apply(
                lambda g: results.embedding.wv.similarity(
                    ds_words[0], g.lower()
                )
            )
        }
    )
    ds_gene_litdistance.sort_values('distance', inplace=True)

    # Dendrogram
    vocabseries = pd.Series(results.embedding.wv.vocab)
    geneselection = vocabseries.index.isin(set(genesymbols.str.lower()))
    geneseries = vocabseries.index[geneselection]
    link = linkage(
        results.embedding.wv.vectors[geneselection, :],
        method='complete', metric='seuclidean'
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Hierarchical Clustering Dendrogram')
    ax.set_xlabel('distance')
    ax.set_ylabel('word')
    tree = dendrogram(
        link,
        # leaf_rotation = 90.,  # rotates the x axis labels
        leaf_font_size=16.,  # font size for the x axis labels
        orientation='left',
        leaf_label_func=lambda v: str(geneseries[v]),
        ax=ax
    )
    geneorder = [geneseries[i] for i in tree['leaves']]
    ds_gene_litdistance['hier_pos'] = ds_gene_litdistance.gene.apply(
        lambda x: geneorder.index(x.lower())
    )
    ds_gene_litdistance['polar_angle'] = (
        ds_gene_litdistance.hier_pos*(2*np.pi-1)/len(ds_gene_litdistance)
    )+.5
    ds_gene_litdistance['polar_r'] = ds_gene_litdistance.distance.apply(
        lambda x: 1-x if x > 0 else 1+x
    )

    # All information now in place to build polar plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    negcor = ds_gene_litdistance.distance < 0
    ax.scatter(
        ds_gene_litdistance.polar_angle[~negcor],
        ds_gene_litdistance.polar_r[~negcor],
        c='g', label='positively associated'
    )
    ax.scatter(
        ds_gene_litdistance.polar_angle[negcor],
        ds_gene_litdistance.polar_r[negcor],
        c='r', label='negatively associated'
    )
    ax.set_rlim([0, 1])
    ax.set_title(f'Gene tm-environment for "{disease_query}"', fontsize=fontsize+5)
    ax.fill_between(
        np.linspace(0, 2*np.pi, 100), 1-sigcorr, 1,
        color='gray', alpha=.5, zorder=10
    )

    ax.annotate(ds_words[0], (0, 0), fontsize=fontsize+5, ha='center')
    annotexts = [
        ax.annotate(
            ds_gene_litdistance.loc[i].gene,
            (ds_gene_litdistance.loc[i].polar_angle, ds_gene_litdistance.loc[i].polar_r),
            fontsize=fontsize
        )
        for i in ds_gene_litdistance[ds_gene_litdistance.distance.abs() >= sigcorr].index
    ]
    if adjustlabels:
        adjust_text(annotexts)  # arrowprops=dict(arrowstyle='->', color='black'))

    ax.legend()

    # Return embedding and plot dataframe
    return results, ds_gene_litdistance
