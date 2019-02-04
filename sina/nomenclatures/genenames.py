# -*- coding: utf-8 -*-
"""Genename nomenclatures for creating text-mining dictionaries
"""
import pandas as pd, numpy as np
from . import makeCIDandTID

def get_gene_dictionary(subset=None):
    """Gene symbol dictionary.
    Uses genenames.org data.

    Args:
        subset (set-like): Set of gene names to subset the dictionary to.
    """
    from bidali.LSD.dealer import genenames
    from bidali.util import unfoldDFlistColumn
    gn = genenames.get_genenames()

    # Subset if requested
    if subset: gn = gn[gn.symbol.isin(subset)].copy()
        
    # Make alias column
    gn['alias'] = gn.T.apply(
        lambda x: [x.symbol, x['name']] + #optionally add other names such as protein names here
        (
            x.alias_symbol.split('|') if x.alias_symbol is not np.nan else []
        )
    )
    print('Original columns',gn.columns)
    gn = gn[['hgnc_id','symbol','alias','uniprot_ids']].copy()
    print('Columns kept:',gn.columns)
    gn = makeCIDandTID(gn)
    return gn

def get_gene_family_dictionary(family_ids):
    """Gene family gene symbol dictionary.
    Uses genenames.org data.
    
    Args:
        family_ids (int or list of ints): Either a single family id, or a list of family ids.

    Example:
        >>> gfd = get_gene_family_dictionary(588) # HLA family
    """
    from bidali.LSD.dealer import genenames
    from bidali.util import unfoldDFlistColumn
    gf = genenames.get_genefamilies()

    # rename some columns
    gf.rename(
        {
            'Approved Symbol': 'symbol',
            'Approved Name': 'name',
            'HGNC ID': 'hgnc_id',
            'Previous Symbols': 'previous'
        }, axis=1, inplace=True
    )
    
    # filter families
    if isinstance(family_ids, int): family_ids = [family_ids]
    gf = gf[gf['Gene family ID'].isin(family_ids)].copy()

    # make alias
    gf['alias'] = gf.T.apply(
        lambda x: [x.symbol, x['name'],]
        +
        (
            x.previous.split(', ') if x.previous is not np.nan else []
        )
        +
        (
            x.Synonyms.split(', ') if x.Synonyms is not np.nan else []
        )
    )
    print('Original columns', gf.columns)
    gf = gf[['hgnc_id','symbol','alias']].copy()
    print('Columns kept:',gf.columns)
    gf = makeCIDandTID(gf)
    return gf

def get_biomart_gene_dictionary():
    from bidali.LSD.dealer.ensembl import get_biomart
    from bidali.genenames import fetchAliases
    bm = get_biomart(atts=[
        'ensembl_gene_id',
        'ensembl_peptide_id',
        'pdb',
        'entrezgene',
        'hgnc_symbol',
    ])
    HGNC_aliases = bm['HGNC symbol'].apply(lambda x: [(a,x) for a in fetchAliases(x,unknown_action='list')])
    HGNC_aliases = pd.concat([pd.Series(dict(a)) for a in HGNC_aliases])
    HGNC_aliases = pd.Series(HGNC_aliases.index.values, index=HGNC_aliases)
    HGNC_aliases.index.set_names('HGNC symbol', inplace = True)
    HGNC_aliases.name = 'HGNC alias'
    return bm.join(other = HGNC_aliases, on = 'HGNC symbol')
