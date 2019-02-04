# -*- coding: utf-8 -*-
"""Ontologies from BioPortal

Reference: https://bioportal.bioontology.org/
"""
from sina.config import secrets
import os, gzip, pandas as pd, requests
from pprint import pprint
from io import TextIOWrapper, StringIO
import urllib.error, urllib.parse
import json

BIOPORTAL_REST_URL = "http://data.bioontology.org"

def get_ontologies(printout=False):
    #need to register -> then https://bioportal.bioontology.org/account API key
    headers = {'Authorization': f"apikey token={secrets.getsecret('bioportal')}" }

    # Get the available resources
    resources = requests.get(
        BIOPORTAL_REST_URL,
        headers = headers
    ).json()
    
    # Get the ontologies from the `ontologies` link
    ontologies = {
        o['acronym']: o
        for o in requests.get(resources["links"]["ontologies"], headers=headers).json()
    }
        
    # Get the name and ontology id from the returned list
    if printout:
        for o in ontologies:
            print(o,ontologies[o]['name'],ontologies[o]['@id'])
    
    else: return ontologies

def print_labels(acronym):
    """
    Reference:
        https://github.com/ncbo/ncbo_rest_sample_code/blob/master/python/python3/get_labels.py

    Example:
        print_labels('PDO')
    """
    headers = {'Authorization': f"apikey token={secrets.getsecret('bioportal')}" }
    
    
    # Get all ontologies from the REST service
    ontology = get_ontologies()[acronym]
    
    labels = []
    
    # Using the hypermedia link called `classes`, get the first page
    page = requests.get(ontology["links"]["classes"], headers=headers).json()
    
    # Iterate over the available pages adding labels from all classes
    # When we hit the last page, the while loop will exit
    while page:
        for ont_class in page["collection"]:
            labels.append(ont_class["prefLabel"])
        if page["links"]["nextPage"]:
            page = requests.get(ontology["links"]["classes"], headers=headers).json()
    
    # Output the labels
    for label in labels:
        print(label)
    
class Ontology(object):
    """Bioportal ontology interface
    
    Args:
        acronym (str): Ontology acronym that will be retrieved.

    TODO:
        - cache result
    """
    def __init__(self, acronym):
        self.acronym = acronym
        self.__ontology = get_ontologies()[self.acronym]
        self.__headers = {'Authorization': f"apikey token={secrets.getsecret('bioportal')}" }
        self.classes = None

    def request(self, url, returnJSON=True):
        """Request url

        Args:
            url (str): Url string.
        """
        r = requests.get(url, headers = self.__headers)
        return r.json() if returnJSON else r

    def get_latest_submission_date(self):
        """Retrieves date when latest submission was created.
        String returned that can be used in file names.
        """
        dateContent = self.request(
            self.__ontology['links']['latest_submission']
        )
        return dateContent['creationDate'][:10].replace('-','_')
    
    def class_page_generator(self):
        """Yield a page of classes
        for the given ontology
        """
        # Yield first page
        page = self.request(self.__ontology["links"]["classes"])
        yield page
        # Yield all following pages
        while page["links"]["nextPage"]:
            page = self.request(page["links"]["nextPage"])
            yield page

    def class_generator(self):
        """Yield the classes
        for the given ontology
        """
        for page in self.class_page_generator():
            yield from page['collection']

    def get_classes(self):
        """Returns all the ontology classes.
        If not previously retrieved, sets attribute self.classes.
        """
        if not self.classes:
            classes = [
                {
                    'name': c['prefLabel'],
                    'source': c['@id'],
                    # alias: all the names starting with prefLabel
                    'alias': [c['prefLabel']]+list(set(c['synonym'])-{c['prefLabel']}),
                    'definition': ' '.join(c['definition'])
                }
                for c in self.class_generator()
            ]
            self.classes = pd.DataFrame(
                classes, columns = ['name', 'alias', 'definition','source']
            )
            duplicates = sum(self.classes['name'].duplicated())
            if duplicates:
                import warnings
                warnings.warn(f'{self.acronym} is not a consistent ontology as it contains duplicated names ({duplicates}).')
                print(self.classes[self.classes['name'].duplicated()].head(10))
                self.classes.drop_duplicates('name',inplace=True)
        return self.classes

    def get_textmining_dict(self,to_csv=False,csv_name_insert_date=True):
        """Returns a pd.DataFrame with concept and term Ids set (CID and TID)
        for use as a text mining dictionary.

        Args:
            to_csv (str): File path. If provided dicts are written to csv file and not returned.
            csv_name_insert_date (bool): If True and writing out, then add date
              of latest submission to the filename.
        """
        from sina.nomenclatures import makeCIDandTID
        tmd = makeCIDandTID(self.get_classes(),conceptCol='name')
        if to_csv:
            csvfilename = (
                f'{to_csv.replace(".csv","")}_{self.get_latest_submission_date()}.csv'
                if csv_name_insert_date else to_csv
            )
            tmd.to_csv(csvfilename)
        else:
            return tmd
