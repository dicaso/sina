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
        rank_column (str | int): column name or number containing the gene rank values;
          the values in this column should either be continuously ascending or descending
          with the most significant values at the top of the table/file.
        **kwargs: are passed through to the pandas `gene_table_file` loading function.

    Example:
        >>> from sina.ranking.ligsea import Ligsea
        >>> lg = Ligsea('neuroblastoma', 'metasta', '/tmp/mock.csv', 'gene_col')
        >>> lg.retrieve_associations()
        >>> lg.determine_gene_associations()
    """
    def __init__(self, topic_query, assoc_regex, gene_table_file, gene_column, rank_column='ranks', assoc_regex_flags=re.IGNORECASE, **kwargs):
        self.topic = topic_query
        self.assoc = (
            assoc_regex if not isinstance(assoc_regex, str) else
            re.compile(assoc_regex, assoc_regex_flags)
        )
        if gene_table_file.endswith('.csv'):
            self.genetable = pd.read_csv(gene_table_file,**kwargs)
        elif gene_table_file.endswith('.xlx') | gene_table_file.endswith('.xlsx'):
            self.genetable = pd.read_excel(gene_table_file,**kwargs)
        else: raise Exception('filetype "%s" not recognised' % gene_table_file.split('.')[-1])
        self.genecol = gene_column
        self.rankcol = rank_column

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
                        # Only looking up if gene symbol if it is not likely to be a general English word
                        gene_symbol = (
                            None if (token.text.isalpha() and (token.is_sent_start or token.text.islower()))
                            else self.get_gene_symbol(token.text)
                        )
                        #if gene_symbol:
                        #    import pdb; pdb.set_trace()
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
                                for gs in gene_symbol:
                                    if not gs in self.gene_association: self.gene_association[gs] = {}
                                    if (association['pmid'],association['date']) not in self.gene_association[gs]:
                                        self.gene_association[gs][(association['pmid'],association['date'])] = []
                                    self.gene_association[gs][(association['pmid'],association['date'])].append(
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

    def get_gene_aliases(self,gene_symbol):
        return list(self.gene_dict[self.gene_dict.symbol == gene_symbol].alias)

    def evaluate_gene_associations(self,infer=False):
        """Curate gene associations made by indicating
        if they are relevant or not

        Args:
            infer (float): A simple machine learning model
              predicts at each annotation made what the next
              annotation will be (online learning), after reaching
              the `infer` float threshold furhter evaluations do not
              have to be provided anymore (TODO work in progress)
        """
        from plumbum import colors
        print(colors.red | 'Type "?" to see the abstract, "!" to skip gene, "x" if good association, anything else if poor association.')
        print(len(self.gene_association),'to evaluate.')
        for geni,gene in enumerate(self.gene_association):
            print(colors.green | 'Reviewing gene "%s" (%s)[%s]:' % (gene,', '.join(self.get_gene_aliases(gene)),geni))
            print(len(self.gene_association[gene]),'associated abstracts.')
            skipGene = False
            for assoc in self.gene_association[gene]:
                for sent_assoc in self.gene_association[gene][assoc]:
                    if skipGene:
                        sent_assoc['valid_annot'] = False
                    else:
                        print(self.gene_association_sents[sent_assoc['sent']])
                        feedback = input()
                        if feedback == '?':
                            print('abstract') #TODO if ? show abstract
                            feedback = input()
                        elif feedback == '!': skipGene = True
                        sent_assoc['valid_annot'] = feedback == 'x'
        self.curated_gene_associations = [
            (gene,assoc,sent_assoc)
            for gene in self.gene_association
            for assoc in self.gene_association[gene]
            for sent_assoc in self.gene_association[gene][assoc]
            if sent_assoc['valid_annot']
        ]
        # Sort according to oldest to newest date
        self.curated_gene_associations.sort(key=lambda x: x[1][1])
        self.curated_gene_associations = pd.DataFrame({
            'gene':[c[0] for c in self.curated_gene_associations],
            'date':[c[1][1] for c in self.curated_gene_associations],
            'pmid':[c[1][0] for c in self.curated_gene_associations],
            'sent':[c[2]['sent'] for c in self.curated_gene_associations],
            'featurevec':[c[2] for c in self.curated_gene_associations]
        })

    def plot_ranked_gene_associations(self):
        import matplotlib.pyplot as plt
        self.curated_gene_associations['ranks'] = [
            self.genetable[self.rankcol][g] for g in self.curated_gene_associations.gene
        ]
        fig,ax = plt.subplots()
        ax.scatter(
            self.curated_gene_associations.date.values,
            self.curated_gene_associations.ranks.values
        )
        ax.set_ylim((self.genetable[self.rankcol].min(),self.genetable[self.rankcol].max()))

    def calculate_nulldistro(self, n, nulldistrosize):
        """Calculate a nulldistribution from the genetable ranks

        Args:
            n (int): Number of elements to sum
            nulldistrosize (int): Number of permutations

        TODO allow for random seed, allow for other function than ranksum
        """
        random_permutation_results = []
        rankvalues = self.genetable[self.rankcol].copy()
        for i in range(nulldistrosize):
            random_permutation_results.append(
                rankvalues.sample(n).sum()
            )
        return pd.Series(random_permutation_results).sort_values()

    def calculate_enrichment(self,rel_alpha=.05,ascending=None,nulldistrosize=1000,max_enrich=None):
        """Caclulate enrichment set for each time point

        Args:
            rel_alpha (float): relevance alpha, level at which set enrichment is
              calculated for least unknown to discover gene
            ascending (bool): if given determines whether rank values are ascending
              or descending with gene significance; if not provided the table is
              interrogated wheter it contains ascending or descending values.
            nulldistrosize (int): the number of permutations to calculate the 
              nulldistributions.
            TODO max_enrich (int): given a maximum number of expected genes involved
              in the process (e.g. by calculating with `predict_number_of_relevant_genes`)
              an enrichment should not go below the least ranked gene by a number at which
              the density of geneset genes and other genes becomes lower than that of the
              originally found enriched set.
        """
        # Select only first mentions of gene associations
        assoc_data = self.curated_gene_associations[~self.curated_gene_associations.gene.duplicated()]
        # Set rank type
        ascending_ranks = (
            self.genetable[self.rankcol].is_monotonic_increasing
            if ascending is None else ascending
        )
        if not hasattr(self,'nulldistros'):
            self.nulldistros = {}
        date_relevancies = {}
        date_relevancies_known_gene = {}
        for date in assoc_data.date.drop_duplicates():
            stratum = assoc_data[assoc_data.date<=date].copy()
            #import pdb; pdb.set_trace()
            # Sort stratum according to ranks
            stratum.sort_values('ranks', ascending=ascending_ranks, inplace=True)
            # Set flag if a relevancy segment has been found
            previousTopPassed = False
            for top in stratum.ranks.drop_duplicates():
                stratum_top = stratum[(stratum.ranks<=top) if ascending_ranks else (stratum.ranks>=top)]
                # Calculate null_distributions
                stratum_top_size = len(stratum_top)
                if stratum_top_size not in self.nulldistros:
                    self.nulldistros[stratum_top_size] = self.calculate_nulldistro(stratum_top_size, nulldistrosize=nulldistrosize)
                # Compare stratum top ranksum
                ranksum = stratum_top.ranks.sum()
                rankprob = (
                    self.nulldistros[stratum_top_size]<=ranksum if ascending_ranks else
                    self.nulldistros[stratum_top_size]>=ranksum
                ).mean()
                if rankprob <= rel_alpha:
                    previousTopPassed = (stratum_top_size, ranksum, rankprob)
                else: break # break if ranksum top of stratum does not meet relevancy cutoff
            if previousTopPassed:
                # Save relevant known gene
                date_relevancies_known_gene[date] = stratum.gene[stratum.index[previousTopPassed[0]-1]]
                # Given that we have a relevant top, we now look for a next 'unknown' gene with combined ranksum value still under cutoff
                stratum_top_size = previousTopPassed[0]
                if stratum_top_size+1 not in self.nulldistros:
                    self.nulldistros[stratum_top_size+1] = self.calculate_nulldistro(stratum_top_size+1, nulldistrosize=nulldistrosize)
                lastTopGene = stratum.gene.iloc[stratum_top_size-1]
                lastTopGene_i = self.genetable.index.get_loc(lastTopGene)
                nextGene = self.genetable.index[lastTopGene_i+1]
                for i,rankvalue in enumerate(self.genetable[self.rankcol].loc[nextGene:]):
                    ranksum = previousTopPassed[1]+rankvalue
                    rankprob = (
                        self.nulldistros[stratum_top_size+1]<=ranksum if ascending_ranks else
                        self.nulldistros[stratum_top_size+1]>=ranksum
                    ).mean()
                    if rankprob > rel_alpha: break
                if i-1 < 0:
                    date_relevancies[date] = lastTopGene # relevancy section contains the last top gene
                else:
                    date_relevancies[date] = self.genetable.index[lastTopGene_i+i]
            else: date_relevancies[date] = None
        
        self.date_relevancies = pd.DataFrame(list(date_relevancies.items()),columns=('date','gene'))
        self.date_relevancies['ranks'] = self.date_relevancies.gene.apply(
            lambda x: None if x is None else self.genetable[self.rankcol][x]
        )
        self.date_relevancies_known_gene = pd.DataFrame(list(date_relevancies_known_gene.items()),columns=('date','gene'))
        self.date_relevancies_known_gene['ranks'] = self.date_relevancies_known_gene.gene.apply(
            lambda x: None if x is None else self.genetable[self.rankcol][x]
        )
        
    def predict_number_of_relevant_genes(self,plot=True,surges=1):
        """Predict number of relevant genes
        as they have been discovered through time.

        Regression with generalised logistic function, info see
        https://en.wikipedia.org/wiki/Generalised_logistic_function

        Args:
            plot (bool): If plotting is not necessary, it can be disabled.
            surges (int): Number of periods of exponential-like discoveries
        """
        import numpy as np
        from scipy.optimize import curve_fit
        import matplotlib.pyplot as plt
        assoc_data = self.curated_gene_associations[~self.curated_gene_associations.gene.duplicated()].copy()
        assoc_data['event'] = range(1,len(assoc_data)+1)
        assoc_data = assoc_data[['date','event']].drop_duplicates('date', keep='last')
        if plot:
            fig,ax = plt.subplots()
            ax.scatter(assoc_data.date.values, assoc_data.event)
        xdata = assoc_data.date.apply(lambda x: x.timestamp()).values
        xdata_norm = (xdata - xdata.min())/xdata.max()
        ydata = assoc_data.event.values
        if surges == 1:
            def sigmoid_func(t, A, K, C, Q, B, M, v):
                return A + (K-A)/((C + Q*np.exp(-B*(t-M)))**(1/v))
            p0 = [ydata.min(), ydata.max(), 1, 1, 1, .0001, 1]
            bounds = (0,np.inf)
        elif surges == 2:
            def sigmoid_func(t, A1, K1, C1, Q1, B1, M1, v1, K2, C2, Q2, B2, M2, v2):
                # M2 stands for the second surge switch time, in which another,
                # linked logistic regression is calculated
                t1_seg = A1 + (K1-A1)/((C1 + Q1*np.exp(-B1*(t[t<M2]-M1)))**(1/v1))
                t2_seg = K1 + (K2-K1)/((C2 + Q2*np.exp(-B2*(t[t>=M2]-M2)))**(1/v2))
                return np.append(t1_seg,t2_seg)
            p0 = [
                ydata.min(), ydata.max()/2, 1, 1, 1, .0001, 1,
                ydata.max()/2, 1, 1, 1, .5, 1
            ]
            bounds = (0,np.inf)
        popt, pcov = curve_fit(sigmoid_func, xdata_norm, ydata, p0=p0, bounds=bounds)
        total_expected = popt[0]+popt[1] if surges==1 else popt[0]+popt[1]+popt[7] # A+K if surges==1 else A1+K1+K2
        if plot:
            ax.plot(assoc_data.date.values, sigmoid_func(xdata_norm, *popt), 'r-', label='fit')
            ax.axhline(total_expected,c='r')
        return total_expected
