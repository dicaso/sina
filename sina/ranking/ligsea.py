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

    Example:
        >>> from sina.ranking.ligsea import Ligsea
        >>> lg = Ligsea('neuroblastoma', 'metasta', '/tmp/mock.csv', 'gene_col')
        >>> lg.retrieve_associations()
        >>> lg.determine_gene_associations()
    """
    def __init__(self, topic_query, assoc_regex, gene_table_file, gene_column, assoc_regex_flags=re.IGNORECASE, **kwargs):
        self.topic = topic_query
        self.assoc = (
            assoc_regex if isinstance(assoc_regex, re.Pattern) else
            re.compile(assoc_regex, assoc_regex_flags)
        )
        if gene_table_file.endswith('.csv'):
            self.genetable = pd.read_csv(gene_table_file,**kwargs)
        elif gene_table_file.endswith('.xlx') | gene_table_file.endswith('.xlsx'):
            self.genetable = pd.read_excel(gene_table_file,**kwargs)
        else: raise Exception('filetype "%s" not recognised' % gene_table_file.split('.')[-1])
        self.genecol = gene_column

    def retrieve_associations(self, corpus_class=PubmedCollection, corpus_location='~/pubmed'):
        """Retrieve associations within the specified `corpus_class`.

        Args:
            corpus_class (sina.documents.BaseDocumentCollection): the document 
              class in which the search will be executed.
            corpus_location (str): Path to corpus directory.
        """
        corpus = corpus_class('pubmed',corpus_location)
        self.documents = corpus.query_document_index(self.topic)
        self.associations = [d for d in self.documents if self.assoc.search(d['content'])]

    def determine_gene_associations(self,verbed=True,twosents=False):
        """Determine sentences with specified gene association
        using natural language processing

        Args:
            verbed (bool): A verb is required in the middle of the regex assoc
              and the gene; this eliminates sentences that do not make a claim 
              on the gene.
            twosents (bool): Lookup possible co-occurence in a sliding
              window of two sentences instead of sentence by sentence
              TODO not implemented yet
        """
        import spacy
        if twosents: raise NotImplementedError
        nlp = spacy.load('en')
        self.gene_association = {}
        self.gene_association_sents = {}
        pos_of_interest = ('VERB', 'NOUN', 'ADP', 'PUNCT', 'GENE')
        for association in self.associations:
            abstract = nlp(association['content'])
            sentences = list(abstract.sents)
            for sent in sentences:
                assoc_match = self.assoc.search(sent.text)
                if assoc_match:
                    sent_startposition = sent[0].idx
                    before_assoc_match = True
                    inbetween_feature_vectors = {}
                    for token in sent:
                        # First check if still before match
                        if (assoc_match.start() < token.idx - sent_startposition) and before_assoc_match:
                            before_assoc_match = False
                            #Store before_assoc_match featurevectors
                            for iv in inbetween_feature_vectors:
                                if not iv in self.gene_association: self.gene_association[iv] = {}
                                if (association['pmid'],association['date']) not in self.gene_association[iv]:
                                    self.gene_association[iv][(association['pmid'],association['date'])] = []
                                self.gene_association[iv][(association['pmid'],association['date'])].append(
                                    inbetween_feature_vectors[iv]
                                )
                            inbetween_feature_vector = {p:0 for p in pos_of_interest}
                            inbetween_feature_vector['sent'] = hash(sent)
                        gene_symbol = self.get_gene_symbol(token.text)
                        if before_assoc_match:
                            if gene_symbol:
                                # For previous genes update GENE count (TODO retroactive for genes coming after)
                                for iv in inbetween_feature_vectors:
                                    inbetween_feature_vectors[iv]['GENE']+=1
                                # Initialise feature vector for each gene symbol
                                for gs in gene_symbol:
                                    inbetween_feature_vectors[gs] = {p:0 for p in pos_of_interest}
                                    inbetween_feature_vectors[gs]['sent'] = hash(sent)
                                    self.gene_association_sents[hash(sent)] = sent
                            elif token.pos_ in pos_of_interest:
                                for iv in inbetween_feature_vectors:
                                    inbetween_feature_vectors[iv][token.pos_]+=1
                        else:
                            if gene_symbol:
                                if not iv in self.gene_association: self.gene_association[iv] = {}
                                if (association['pmid'],association['date']) not in self.gene_association[iv]:
                                    self.gene_association[iv][(association['pmid'],association['date'])] = []
                                self.gene_association[iv][(association['pmid'],association['date'])].append(
                                    inbetween_feature_vector.copy()
                                )
                                self.gene_association_sents[hash(sent)] = sent
                                inbetween_feature_vector['GENE']+=1
                            elif token.pos_ in pos_of_interest:
                                inbetween_feature_vector[token.pos_]+=1

    def get_gene_symbol(self,token):
        """If token is a gene synonym, return reference gene symbol,
        otherwise returns None.
        
        Current implementation only for human gene names.

        Args:
            token (str): The token str
        """
        if not hasattr(self, 'gene_dict'):
            from sina.nomenclatures.genenames import get_gene_dictionary
            self.gene_dict = get_gene_dictionary()
            #Normalize alias names to lower case
            self.gene_dict.index = self.gene_dict.alias.str.lower()
        if token.lower() in self.gene_dict.index:
            result = self.gene_dict.loc[token.lower()].symbol
            return (result,) if isinstance(result,str) else tuple(result)
        else: return None
