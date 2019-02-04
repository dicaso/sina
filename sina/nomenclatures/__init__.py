# -*- coding: utf-8 -*-
"""Nomenclature submodule for creating text-mining dictionaries
and working with ontologies.
"""
import pandas as pd, numpy as np

# Utilities for making dictionaries
##CIDcounter class to create CID (concept id) according to AS requirement that it
##is the same as TID reference symbol
class CIDcounter:
    def __init__(self,startvalue=0):
        import itertools as it
        self.counter = it.count(startvalue)
        
    def next(self,increment):
        if increment:
            self.value = next(self.counter)
        else: next(self.counter) #this step ensures a CID == TID for a new reference concept
        return self.value

def makeCIDandTID(df,conceptCol='symbol',aliasCol='alias'):
    """Make text mining dictionary with TID and CID set

    According to Adil Salhi's text mining requirements.

    Args:
        df (pd.DataFrame): Dataframe to transform.
        conceptCol (str): Column name of main concept/symbol.
        aliasCol (str): Column name where all aliases for the concept
          are gathered in a list, including the concept itself as first list member.
    """
    from bidali.util import unfoldDFlistColumn
    df = unfoldDFlistColumn(df,aliasCol)
    df.reset_index(inplace=True, drop=True)
    df.index.name = 'TID' #term id
    c = CIDcounter()        
    df.insert(0, 'CID', (~df[conceptCol].duplicated()).apply(c.next)) #need to work with `not` duplicated column!
    return df
