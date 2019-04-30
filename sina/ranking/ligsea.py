# -*- coding: utf-8 -*-
"""Literature geneset enrichment analysis module

Processes an experimentally ranked list and makes a comparison
with the liturature in a certain topical field and association with
a keyphrase noun chunk.
"""
import re, pandas as pd, numpy as np
from sina.documents import PubmedCollection
from collections import OrderedDict

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
        >>> lg.retrieve_documents()
        >>> lg.determine_gene_associations() #issue downloading genenames
        >>> lg.evaluate_gene_associations()
        >>> lg.predict_number_of_relevant_genes(surges=1)
        >>> lg.plot_ranked_gene_associations()
        >>> lg.calculate_enrichment()
        >>> lg.retrieve_datasets()
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

    def to_json(self, filename, attributes={'documents','associations','gene_association','gene_association_sents'}):
        """Save object state to json, by default all documents and attributes provided in
        extra_attributes.
        
        Args:
            attributes (set): the set of attributes to save.

        TODO: does not work yet for gene_association, because it has tuple keys
        """
        import json, os
        class DTEncoder(json.JSONEncoder): 
            def default(self, o):
                if isinstance(o,datetime.datetime):
                    return ('timestamp', o.timestamp())
                else:
                    return super().default(o)
                    
        json.dump(
            {
                attr: getattr(self, attr)
                for attr in set(attributes)&set(dir(self))
            },
            os.path.expanduser(filename),
            cls = DTEncoder
        )

    def from_json(self):
        """Load in object from json file"""
        raise NotImplementedError

    def retrieve_documents(self, corpus_class=PubmedCollection, corpus_location='~/pubmed'):
        """Retrieve associations within the specified `corpus_class`.

        Args:
            corpus_class (sina.documents.BaseDocumentCollection): the document 
              class in which the search will be executed.
            corpus_location (str): Path to corpus directory.
        """
        self.corpus = corpus_class('pubmed',corpus_location)
        self.documents = self.corpus.query_document_index(self.topic)
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
        try: nlp = spacy.load('en')
        except OSError:
            raise Exception(
                '''spacy language module not installed.
                Run: python -m spacy download en
                '''
            )
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
                        # Only looking up if gene symbol if it is not likely to be a general English word
                        gene_symbol = (
                            None if (token.text.isalpha() and (token.is_sent_start or token.text.islower()))
                            else self.get_gene_symbol(token.text)
                        )
                        if gene_symbol:
                        #    import pdb; pdb.set_trace()
                            association_key = (
                                association['pmid'],association['date'],token.text,(assoc_match.start(),assoc_match.end())
                            )
                        else:
                            association_key = None
                        # First check if still before match
                        if (assoc_match.start() < token.idx - sent_startposition) and before_assoc_match:
                            before_assoc_match = False
                            #Store before_assoc_match featurevectors
                            for iv in inbetween_feature_vectors:
                                if not iv in self.gene_association: self.gene_association[iv] = {}
                                prev_association_key = inbetween_feature_vectors[iv].pop('association_key')
                                if prev_association_key not in self.gene_association[iv]:
                                    self.gene_association[iv][prev_association_key] = []
                                self.gene_association[iv][prev_association_key].append(
                                    inbetween_feature_vectors[iv]
                                )
                            inbetween_feature_vector = {p:0 for p in pos_of_interest}
                            inbetween_feature_vector['sent'] = hash(sent)
                        if before_assoc_match:
                            if gene_symbol:
                                # For previous genes update GENE count (TODO retroactive for genes coming after)
                                for iv in inbetween_feature_vectors:
                                    inbetween_feature_vectors[iv]['GENE']+=1
                                # Initialise feature vector for each gene symbol
                                for gs in gene_symbol:
                                    inbetween_feature_vectors[gs] = {p:0 for p in pos_of_interest}
                                    inbetween_feature_vectors[gs]['sent'] = hash(sent)
                                    inbetween_feature_vectors[gs]['association_key'] = association_key
                                    self.gene_association_sents[hash(sent)] = sent
                            elif token.pos_ in pos_of_interest:
                                for iv in inbetween_feature_vectors:
                                    inbetween_feature_vectors[iv][token.pos_]+=1
                        else:
                            if gene_symbol:
                                for gs in gene_symbol:
                                    if not gs in self.gene_association: self.gene_association[gs] = {}
                                    if association_key not in self.gene_association[gs]:
                                        self.gene_association[gs][association_key] = []
                                    self.gene_association[gs][association_key].append(
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

    def evaluate_gene_associations(self,infer=False,store=True,revaluate=False,server=False):
        """Curate gene associations made by indicating
        if they are relevant or not

        Args:
            infer (float): A simple machine learning model
              predicts at each annotation made what the next
              annotation will be (online learning), after reaching
              the `infer` float threshold further evaluations do not
              have to be provided anymore (TODO work in progress)
            store (bool): Annotations are stored in corpus directory
            revaluate (bool): Previous annotations are not retrieved
            server (bool | str): Evaluate through browser, instead of commandline.
              If str is provided is passed on to server function as address, e.g. '0.0.0.0'
        """
        import shelve, hashlib, os

        # Web ui
        if server:
            import itertools as it
            c1,c2 = it.count(),it.count()
            anse = AnnotaterServer(
                sentences = [
                    self.gene_association_sents[sent_assoc['sent']].text
                    for gene in self.gene_association
                    for assoc in self.gene_association[gene]
                    for sent_assoc in self.gene_association[gene][assoc]
                ],
                annotations = [
                    ['',{
                        0: {
                            'label': assoc[2],
                            'id': 0,
                            'parent_id': next(c1),
                            'tag_annotations': 'gene'
                            },
                        1: {
                            'label': self.gene_association_sents[
                                sent_assoc['sent']
                                ].text[assoc[3][0]:assoc[3][1]],
                            'id': 1,
                            'parent_id': next(c2),
                            'tag_annotations': self.assoc.pattern
                            }
                    }]
                    for gene in self.gene_association
                    for assoc in self.gene_association[gene]
                    for sent_assoc in self.gene_association[gene][assoc]
                ],
                tags = ['gene',self.assoc.pattern,'relation'],
                host = server if isinstance(server,str) else '127.0.0.1'
            )
            anse.run()

        # CLI
        else:
            from plumbum import colors
            print(
                colors.red | (
                    'Type "?" to see the abstract, "!" to skip gene alias and "!!" to skip entire gene,\n'+
                    '"+" if strong positive association, "+-" if weak positive association,\n'+
                    '"-" for strong negative association, "-+" if weak negative association,\n'+
                    '"x" for unclear association, anything else if invalid association.'
                )
            )
            print(len(self.gene_association),'to evaluate.')
            with shelve.open(os.path.join(os.path.expanduser(self.corpus_location),'.annotations.shelve')) as stored_annots:
                for geni,gene in enumerate(self.gene_association):
                    print(colors.green | 'Reviewing gene "%s" (%s)[%s]:' % (gene,', '.join(self.get_gene_aliases(gene)),geni))
                    print(len(self.gene_association[gene]),'associated abstracts.')
                    skipGene = False
                    skipAliases = set()
                    for assoc in self.gene_association[gene]:
                        for sent_assoc in self.gene_association[gene][assoc]:
                            sent_store_key = str((
                                hashlib.md5(self.gene_association_sents[sent_assoc['sent']].text.encode()).hexdigest(),
                                gene, assoc[2], assoc[3] # gene alias and regex match positions
                            ))
                            if not revaluate and sent_store_key in stored_annots:
                                sent_assoc['valid_annot'] = stored_annots[sent_store_key]
                                continue
                            if skipGene or assoc[2] in skipAliases:
                                sent_assoc['valid_annot'] = False
                            else:
                                s = self.gene_association_sents[sent_assoc['sent']]
                                print(
                                    s.text.replace(
                                        s.text[assoc[3][0]:assoc[3][1]],colors.red | s.text[assoc[3][0]:assoc[3][1]]
                                    ).replace(
                                        assoc[2], colors.green | assoc[2]
                                    )
                                )
                                feedback = input()
                                if feedback == '?':
                                    print('abstract')
                                    feedback = input()
                                elif feedback == '!': skipAliases.add(assoc[2])
                                elif feedback == '!!': skipGene = True
                                sent_assoc['valid_annot'] = feedback if feedback in ('+', '+-', '-', '-+', 'x') else False
                            if store: stored_annots[sent_store_key] = sent_assoc['valid_annot']
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
                'featurevec':[c[2] for c in self.curated_gene_associations],
                'annot': [c[2]['valid_annot'] for c in self.curated_gene_associations]
            })
    
    def train_gene_evaluations(self, test_size=0.25, random_state=1000):
        """Based on evaluations already provided
        build an nlp model to either evaluate associations
        not yet reviewed, or get an estimate of how consistent
        the evaluations seem to be.

        Args:
            test_size (float): Percentage of data for testing.
            random_state (float): Random seed.

        Reference:
            https://realpython.com/python-keras-text-classification/
        """
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        df = pd.DataFrame(
            [
                (
                self.gene_association_sents[s_entry['sent']].text,
                s_entry['valid_annot'] if 'valid_annot' in s_entry else None,
                gene
                )
            for gene in self.gene_association
            for entry in self.gene_association[gene]
            for s_entry in self.gene_association[gene][entry]
            ], columns=['sentence', 'label', 'source']
        )
        
        # Prepare training/test data
        # Sort df on maximum annotation (sentences that have one strong meaningful annotation
        # will be retained after removing the duplicates)
        df['label_int'] = df.label.map(
            {'x': 0, False: 0, '+-': 1, '+': 2, '-+': -1, '-': -2}
        )
        df['label_int_abs'] = df.label_int.abs()
        df.sort_values('label_int_abs', inplace=True, ascending=False)
        sentences = df[~df.sentence.duplicated()].dropna().sentence.values #drop unannotated
        y = df[~df.sentence.duplicated()].dropna().label_int.values
        sentences_train, sentences_test, y_train, y_test = train_test_split(
            sentences, y, test_size=test_size, random_state=random_state
        )

        # Vectorize
        vectorizer = CountVectorizer(min_df=0, lowercase=False, max_features=10000)
        vectorizer.fit(sentences_train)
        X_train = vectorizer.transform(sentences_train)
        X_test  = vectorizer.transform(sentences_test)

        # Logistic regression
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        print("Logistic regression accuracy:", score)

        # Neural network classification
        from keras.models import Sequential
        from keras import layers
        input_dim = X_train.shape[1]
        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',  
            optimizer='adam',  
            metrics=['accuracy']
        )
        print(model.summary())
        history = model.fit(
            X_train, y_train,
            epochs=100,
            verbose=False,
            validation_data=(X_test, y_test),
            batch_size=10
        )
        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

        # Plot
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

    def plot_ranked_gene_associations(self,aggregate=np.mean,geneLines=True):
        """Plot the associations of the ranked genes

        Args:
            aggregate: function to apply if a gene has more than one rank
              value, the function needs to return the same value if only one
              value is provided (e.g. np.mean, np.max, np.min)
            geneLines (bool): draw lines from first mention of a gene.
        """
        import matplotlib.pyplot as plt
        genetable = self.genetable.set_index(self.genecol)
        self.curated_gene_associations['ranks'] = [
            aggregate(genetable[self.rankcol][g]) if g in genetable.index else None
            for g in self.curated_gene_associations.gene
        ]
        fig,ax = plt.subplots()
        first_gene_mention = ~self.curated_gene_associations.gene.duplicated()
        ax.scatter(
            self.curated_gene_associations[first_gene_mention].dropna().date.values,
            self.curated_gene_associations[first_gene_mention].dropna().ranks.values
        )
        if geneLines:
            ax.hlines(
                self.curated_gene_associations[first_gene_mention].dropna().ranks.values,
                self.curated_gene_associations[first_gene_mention].dropna().date.values,
                self.curated_gene_associations[first_gene_mention].dropna().date.max(),
                color = 'b'
            )
        ax.scatter(
            self.curated_gene_associations[~first_gene_mention].dropna().date.values,
            self.curated_gene_associations[~first_gene_mention].dropna().ranks.values
        )
        ax.set_ylim((genetable[self.rankcol].min(),genetable[self.rankcol].max()))
        ax.set_xlabel('publication year')
        ax.set_ylabel('gene rank')
        ax.set_title('Experimentally ranked <%s> associated <%s> genes' % (self.assoc.pattern, self.topic))
        return ax

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

    def calculate_enrichment(
            self,rel_alpha=.05,ascending=None,nulldistrosize=1000,max_enrich=None,
            plot=True, enrich_color='g', enrich_marker='+', enrich_line=True
        ):
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
        # Prepare genetable
        genetable = self.genetable.set_index(self.genecol)
        genetable = genetable[(~genetable.index.isna())&(~genetable.index.duplicated())]
        
        # Select only first mentions of gene associations
        assoc_data = self.curated_gene_associations[~self.curated_gene_associations.gene.duplicated()]
        # Set rank type
        ascending_ranks = (
            genetable[self.rankcol].is_monotonic_increasing
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
                lastTopGene_i = genetable.index.get_loc(lastTopGene)
                nextGene = genetable.index[lastTopGene_i+1]
                for i,rankvalue in enumerate(genetable[self.rankcol].loc[nextGene:]):
                    ranksum = previousTopPassed[1]+rankvalue
                    rankprob = (
                        self.nulldistros[stratum_top_size+1]<=ranksum if ascending_ranks else
                        self.nulldistros[stratum_top_size+1]>=ranksum
                    ).mean()
                    if rankprob > rel_alpha: break
                if i-1 < 0:
                    date_relevancies[date] = lastTopGene # relevancy section contains the last top gene
                else:
                    date_relevancies[date] = genetable.index[lastTopGene_i+i]
            else: date_relevancies[date] = None
        
        self.date_relevancies = pd.DataFrame(list(date_relevancies.items()),columns=('date','gene'))
        self.date_relevancies['ranks'] = self.date_relevancies.gene.apply(
            lambda x: None if x is None else genetable[self.rankcol][x]
        )
        self.date_relevancies_known_gene = pd.DataFrame(list(date_relevancies_known_gene.items()),columns=('date','gene'))
        self.date_relevancies_known_gene['ranks'] = self.date_relevancies_known_gene.gene.apply(
            lambda x: None if x is None else genetable[self.rankcol][x]
        )
        if plot:
            ax = self.plot_ranked_gene_associations()
            plotfn = ax.plot if enrich_line else ax.scatter
            plotfn(
                self.date_relevancies.dropna().date, self.date_relevancies.dropna().ranks,
                c=enrich_color, marker=enrich_marker
            )
        
    def predict_number_of_relevant_genes(self,plot=True,surges=1,predict_future_years=50):
        """Predict number of relevant genes
        as they have been discovered through time.

        Regression with generalised logistic function, info see
        https://en.wikipedia.org/wiki/Generalised_logistic_function

        Args:
            plot (bool): If plotting is not necessary, it can be disabled.
            surges (int): Number of periods of exponential-like discoveries.
            predict_future_years (int): Number of years in the future to show
              fitted curve.
        """
        import datetime
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
            if predict_future_years:
                # Get start date
                unix_epoch = np.datetime64(0, 's')
                one_second = np.timedelta64(1, 's')
                first_discovery_date = datetime.datetime.utcfromtimestamp(
                    (assoc_data.date.values[0] - unix_epoch) / one_second # convert to seconds
                )
                preddates = [
                    first_discovery_date + datetime.timedelta(days=365*y)
                    for y in range(predict_future_years)
                ]
                pred_norm = [(d.timestamp()-xdata.min())/xdata.max() for d in preddates]
                ax.plot(preddates, sigmoid_func(pred_norm, *popt), 'r-', label='fit')
            else:
                ax.plot(assoc_data.date.values, sigmoid_func(xdata_norm, *popt), 'r-', label='fit')
            ax.axhline(total_expected,c='r')
            ax.set_xlabel('publication year')
            ax.set_ylabel('# of genes')
            ax.set_title('<%s> associated <%s> genes' % (self.assoc.pattern, self.topic))
        return total_expected

    def retrieve_datasets(self,only_curated_genes=True):
        import xml.etree.ElementTree as ET
        if only_curated_genes:
            pmids = self.curated_gene_associations.pmid.drop_duplicates()
        else: pmids = [d['pmid'] for d in lg.documents]
        self.article_xmls = self.corpus.retrieve_article_xmls(pmids)
        accarticles = [a for a in self.article_xmls.values() if list(a.iter('AccessionNumberList'))]
        #other interesting xlm elements CoiStatement, DataBank, DataBankList, GeneSymbolList
        geos = []
        for a in accarticles: 
            for anl in a.iter('AccessionNumberList'):
                for an in anl.iter('AccessionNumber'):
                    if an.text.startswith('G'): geos.append(an.text)
        self.geos = geos

class AnnotaterServer(object):
    def __init__(self, sentences, annotations=None, tags=[], host='127.0.0.1'):
        """Annotation server

        References:
            https://www.w3.org/TR/annotation-model/
            https://en.wikipedia.org/wiki/Web_annotation
            http://docs.annotatorjs.org
            https://www.w3.org/TR/selection-api/
        
        Args:
            sentences (list): list of sentence strings to annotate
            annotations (list): list of annotations for every sentence
            tags (list): list of tags that will be preloaded in ui legend
            host (str): host ip
        """
        from flask import Flask, request, render_template
        self.app = Flask('sina') #needs to be package name, where templates/static folder can be found!
        self.host = host
        self.sentences = sentences
        self.annotations = annotations if annotations else OrderedDict()
        self.tags = tags

    def run(self, debug=False):
        from flask import Flask, request, render_template, jsonify

        @self.app.route('/')
        @self.app.route('/<int:sent_id>')
        def index(sent_id=0):
            if sent_id in self.annotations:
                html_sentence = applyAnnotations(
                    self.sentences[sent_id],
                    self.annotations[sent_id]
                )
            else: html_sentence = self.sentences[sent_id]
            return render_template(
                'index.html',
                sentence=html_sentence,
                docid=sent_id,
                prevSent=sent_id-1,
                nextSent=sent_id+1 if sent_id+1<len(self.sentences) else None
            )

        @self.app.route('/api/annotations',methods=['GET','POST'])
        def api():
            self.app.logger.debug(request.json)
            # POST data looks like
            #{'quote': 'there', 'ranges': [{'start': '', 'startOffset': 53, 'end': '', 'endOffset': 58}], 'text': '', 'tags': ['first']}
            if request.json:
                data = request.json
                docid = int(data['parent_id']) if 'parent_id' in data else int(data['id'])
                if not docid in self.annotations:
                    self.annotations[docid] = ['',OrderedDict()]
                if data['type'] == 'segment_annotation':
                    self.annotations[docid][1][data['id']] = data
                elif data['type'] == 'doc_annotation':
                    self.annotations[docid][0] = data['doc_annotations']
                    
                return jsonify({'id': data['id']})

        @self.app.route('/api/update',methods=['POST'])
        def update_annotation():
            self.app.logger.debug(request.json)
            if request.json:
                return jsonify({'id': 0})

        @self.app.route('/api/delete',methods=['DELETE'])
        def remove_annotation():
            self.app.logger.debug(request.json)
            if request.json:
                data = request.json
                docid = int(data['parent_id'])
                self.annotations[docid][1].pop(data['id'])
                return jsonify({'removed_id': data['id']})

        @self.app.route('/api/search/<int:doc_id>/<int:annot_id>',methods=['GET'])
        def search(doc_id,annot_id):
            self.app.logger.debug(request.form)
            return jsonify(
                {
                'id':annot_id,
                'tags':[self.annotations[doc_id][1][annot_id]['tag_annotations']] #TODO tags not a list
                }
            )

        @self.app.route('/api/docannotation/<int:doc_id>',methods=['GET'])
        def get_docannot(doc_id):
            self.app.logger.debug(request.form)
            return jsonify(
                {
                    'id':doc_id,
                    'doc_tags': self.annotations[doc_id][0]
                    if doc_id in self.annotations else ''
                }
            )

        @self.app.route('/api/tags',methods=['GET'])
        def get_tags():
            return jsonify(
                self.tags if self.tags else
                sorted({
                    self.annotations[d][1][a]['tag_annotations'] for d in self.annotations for a in self.annotations[d][1]
                })
            )

        @self.app.route('/quit')
        def shutdown():
            shutdown_hook = request.environ.get('werkzeug.server.shutdown')
            if shutdown_hook is not None:
                shutdown_hook()
            return 'Server shutdown'
        
        self.app.run(host=self.host, debug=debug)

def applyAnnotations(sent,annots):
    replacements = []
    for annot_id,annot in annots[1].items():
        startPos = -1
        for i in range(annot['precedingMatches']+1):
            startPos = sent.index(annot['label'],startPos+1)
        replacements.append((startPos,startPos+len(annot['label']),annot))
    # Sort so that downstream replacements can be handled first
    replacements.sort(key=lambda x: x[0],reverse=True)
    prevSentDifferences = [] # to deal with overlapping annotations
    for i, (startPos,endPos,rplcmnt) in enumerate(replacements):
        prevSent = sent
        endPos += sum([ld for sp,ld in prevSentDifferences if endPos>sp])
        # if current endPos is bigger than a previous applied startPos it should be encompassing
        # that previous annotation
        sent = '{}<annot-8 class="annotator-hl" data-id="{}">{}</annot-8>{}'.format(
            sent[:startPos],
            rplcmnt['id'],
            sent[startPos:endPos],
            sent[endPos:]
        )
        prevSentDifferences.append((startPos,len(sent)-len(prevSent)))
    return sent
