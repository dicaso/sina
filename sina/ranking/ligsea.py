# -*- coding: utf-8 -*-
"""Literature geneset enrichment analysis module

Processes an experimentally ranked list and makes a comparison
with the liturature in a certain topical field and association with
a keyphrase noun chunk.
"""
import re, pandas as pd
from sina.documents import PubmedCollection

class Ligsea(object):
    """Ligsea class that instaniates a literature
    geneset enrichment analysis object

    Args:
        topic_query (str): the query that selects the set of documents that will
          be used for the `ligsea` analysis.
        assoc_regex (str | re.Pattern): the regular expression for which co-expression
          with the gene list will be investigated; can be a str or compiled regex.
        gene_table_file (str): filepath of the ranked gene list, should end in '.csv',
          '.xlx', or '.xlsx', the two recognised file types.
        gene_column (str | int): column name or number containing the gene labels in
          `gene_table_file`.
        **kwargs: are passed through to the pandas `gene_table_file` loading function.
    """
    def __init__(self, topic_query, assoc_regex, gene_table_file, gene_column, **kwargs)
        self.topic = topic_query
        self.assoc = assoc_regex if isinstance(assoc_regex, re.Pattern) else re.compile(assoc_regex)
        if gene_table_file.endswith('.csv'):
            self.genetable = pd.read_csv(gene_table_file,**kwargs)
        elif gene_table_file.endswith('.xlx') | gene_table_file.endswith('.xlsx'):
            self.genetable = pd.read_excel(gene_table_file,**kwargs)
        else: raise Exception('filetype "%s" not recognised' % gene_table_file.split('.')[-1])
        self.genecol = gene_column

    def retrieve_associations(self,corpus_class=PubmedCollection):
        """Retrieve associations within the specified `corpus_class`.

        Args:
            corpus_class (sina.documents.BaseDocumentCollection): the document 
              class in which the search will be executed.
        """
        corpus = corpus_class('pubmed','~/pubmed')
        self.associations = corpus.filter_documents(self.topic_query)
