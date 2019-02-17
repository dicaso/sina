# -*- coding: utf-8 -*-
"""Simulations module

Classes to simulate a gene-centred experiment or series of experiments.
With a full series of experiments over a timecourse a buildup of a
geneset can be tested.
"""

class GeneExperiment(object):
    """GeneExperiment class

    This class defines expectations of a gene-centred experiment,
    some of the assumptions will introduce bias, but it is just a
    model to test how an experiment might produce results giving
    a set of general meta-experiment related assumptions.

    Args:
        geneset (str set | int): set of gene symbols that is considered relevant for
          the experiment. If an `int` is provided a random set of genes of that 
          number is chosen.
        complexity (int): at a theoretical level, the complexity would stand for
          the amount of independent concepts/processes/factors that are at play
          in the experiment; at a practical level, it is here considered to be the
          amount of meaningful principal components, you would obtain performing a
          PCA analysis. In this context a batch effect might be considered, although
          if it could be accounted for in the analysis, it should have been eliminated
          as much as possible in the results that are being considered, and as such would
          no longer play a meaningful role. If set to 0, do not consider complexity in
          the simulation. The size of each "PCA" component is chosen similar to the size
          of the geneset under consideration. Theoretically you could make a more complicated
          model, where the components have different possible sizes, but for now I do not
          see a benefit of adding that to the model.
        noise (float): a float in the range [0 1[, with 0 meaning no added noise and 1 so
          much noise added that the result should be indistinguishable from a random sequence
        resolution (float): the resolution determines how rankable the `experimental` results
          are. From a theoretical perspective, one could expect that different genes would
          always be discernable as to their process importance, but depending on the power
          of the experiment, it might not be resolvable consistently at the interpretation
          level of the results. Should be a float in the range [0 1], 0 only one value in
          the results, essentially meaningless, 1 a different value for each gene.
        complexrnk (int, <=complexity): the complexity rank, indicates with which 
          'PCA component' the geneset would correspond in this experiment.
        relevantset (float): a float in the range ]0 1], with 0 meaning no gene is actually
          relevant in the current experiment, and 1 meaning all genes are relevant, and will
          only be influenced by noise. Non relevant genes will get a random position in the
          ranking, although due to complexity they might also randomly end up in an averagely
          higher ranked group.
        completenes (float): a float in the range ]0 1] indicating how complete the set is,
          in respect of a the theoretical complete number that has not yet been discovered
          in the complete set of literature.
        
    """
    def __init__(
            self, geneset,
            complexity=1, noise=.9, resolution=1,
            complexrnk=1, relevantset=.9, completeset=.9
        ):
        self.settings = {
            'experiment': {
                'complexity': complexity,
                'noise': noise,
                'resolution': resolution
            },
            'geneset': {
                'complexrnk': complexrnk,
                'relevantset': relevantset,
                'completeset': completeset
            }
        }
        if isinstance(geneset, int):
            self.geneset = set(self.genelist.sample(geneset))
        else: self.geneset = geneset

    @property
    def genelist(self):
        cls = type(self)
        if not hasattr(cls, 'genelist_'):
            #TODO loading gene_dictionary is not efficient, should take
            #from other source
            from sina.nomenclatures.genenames import get_gene_dictionary
            cls.genelist_ = get_gene_dictionary().symbol.drop_duplicates().reset_index().symbol
        return cls.genelist_

class GeneExperimentSeries(object):
    """In the GeneExperimentSeries class
    instead of one GeneExperiment, a series of such experiments
    is created, each of which is assigned to a unformly drawn date
    as per the range provided. According to the results of the experiment
    certain gene associations are being made.

    This allows for a full simulation of a literature based search.
    Including possible false negatives, and false positives. Of course
    it is a limited model, in the sense that each 'abstract' is associated
    with just one generated experiment, and functional assays are not being
    performed that more powerfully indicate the possible function of a gene
    in a certain context. This should not be a limitation, if the goal of
    the model is more focused on the general growth dynamics, rather than
    exact science on particular genes. In the experiment series the start
    point is a ground truth set of genes, for which functional assays would
    then be rather pointless.

    Args:
        geneset (set of str): The main geneset
        startDate (str): date str of the start of the period
        endDate (str): date str of the end of the period
        iterations (int): number of experiments created,
          not all iterations will necessarily produce an
          association
        ...
    """
