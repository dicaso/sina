# -*- coding: utf-8 -*-
"""Sina document sources

Everything for setting up document base, such as pubmed, patents, ...

PubmedCollection:
  References:
    - https://www.ncbi.nlm.nih.gov/books/NBK25499/
    - https://www.nlm.nih.gov/databases/download/data_distrib_main.html
    - https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/

  Based on Adil Salhi's initial code and workflow.
  Discussed points for improvement (but depends on making DES downstream changes):
    - now a limited set of publication metadata is extracted, this should be reviewed and if useful expanded
    - publication date is registered to database as string, because of this cannot be indexed and logically searched as date

  Technical improvements:
    - tool and user email should be included in request call
    - ncbi only allows certain number of requests per second
        "api_key – enforced in May 2018
        In May 2018, NCBI will begin enforcing the practice of using an API key for sites that post more than 3 requests per second. 
        Please see Chapter 2 for a full discussion of this new policy."
        https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.Usage_Guidelines_and_Requiremen
    - change pmid request to post
"""
from abc import ABC, abstractmethod
from sina.documents.utils import default_retmax, valid_date
import requests, os, glob, gzip
import xml.sax
import xml.etree.ElementTree as ET

class BaseDocumentCollection(ABC):
    def __init__(self, name, location, remote=None, startTime=None, endTime=None, timeUnit=1):
        self.name = name
        self.location = os.path.expanduser(location)
        self.remote = remote
        self.start = startTime
        self.end = endTime
        self.timeUnit = timeUnit

    @abstractmethod
    def retrieve_documents(self, *args, **kwargs):
        """retrieve documents to the local system"""
        pass
    
    @abstractmethod
    def process_documents(self, callback, *args, **kwargs):
        """process all documents with a callback function"""
        pass

    @abstractmethod
    def filter_documents(self, regex, *args, **kwargs):
        """filter a subset of documents"""
        pass

# Pubmed
## corpus class
class PubmedCollection(BaseDocumentCollection):
    """Example:

    >>> pmc = PubmedCollection('pubmed','~/pubmed')
    >>> pmc.retrieve_documents()
    """
    def retrieve_documents(self):
        import ftplib
        ftp = ftplib.FTP(
            host='ftp.ncbi.nlm.nih.gov',
            user='anonymous',
            passwd='christophe.vanneste@kaust.edu.sa' #TODO fetch from kindi
        )
        ftp.cwd('pubmed/baseline')
        filenames = ftp.nlst()
        for filename in filenames:
            print('Retrieving',filename)
            with open(os.path.join(self.location,filename),'wb') as fh:
                ftp.retrbinary('RETR '+ filename, fh.write)
        ftp.quit()

    def process_documents(self,callback,*args,verbose=False,progress=True,**kwargs):
        """process documents with a callback function
        
        Args:
            callback (function): function that will take the title, abstract text, and full
              xml element as first positional arguments and any further provided arguments
        """
        xmlfilenames = glob.glob(os.path.join(self.location,'*.xml.gz'))
        totalxmlfiles = len(xmlfilenames)
        for xmli,xmlfilename in enumerate(xmlfilenames):
            #TODO check md5
            with gzip.open(xmlfilename) as xmlfilehandler:
                context = iter(ET.iterparse(xmlfilehandler, events=("start", "end")))
                event, root = next(context)
                for event, elem in context:
                    if event == "start":
                        pass
                    elif event == "end" and elem.tag == "PubmedArticle":
                        try:
                            title = elem.find('MedlineCitation/Article/ArticleTitle').text
                            abstract = elem.find('MedlineCitation/Article/Abstract/AbstractText').text
                            callback(title,abstract,elem,*args,**kwargs)
                        except AttributeError as e:
                            if verbose: print(e)
                        #TODO output article
                        root.clear() # ensures only one article is kept in memory
                        # but because of the 'clear' you cannot use the root for info on
                        # current iterated PubmedArticle element
            if progress: print(end='\rProgress (%): {:<10}'.format(xmli/totalxmlfiles)*100)

    def build_document_index(self):
        """Build an index for fast document retrieval"""
        from whoosh.index import create_in
        from whoosh import fields
        schema = fields.Schema(
            title=fields.TEXT(stored=True),
            pmid=fields.ID(stored=True),
            content=fields.TEXT(stored=True),
            date=fields.DATETIME(stored=True)
        )
        indexdir = os.path.join(self.location,'.index')
        os.mkdir(indexdir)
        ix = create_in(indexdir, schema)
        def commit_abstracts(title,abstract,elem):
            import datetime
            date = datetime.datetime(
                int(elem.find('MedlineCitation/DateCompleted/Year').text),
                int(elem.find('MedlineCitation/DateCompleted/Month').text),
                int(elem.find('MedlineCitation/DateCompleted/Day').text)
            )
            writer = ix.writer()
            writer.add_document(
                title=title,
                pmid=elem.find('MedlineCitation/PMID').text,
                content=abstract,
                date=date
            )
            writer.commit()
        self.process_documents(commit_abstracts)

    def query_document_index(self,query):
        import whoosh.index as index
        from whoosh.qparser import QueryParser
        indexdir = os.path.join(self.location,'.index')
        ix = index.open_dir(indexdir)
        with ix.searcher() as searcher:
            query = QueryParser("content", ix.schema).parse(query)
            results = searcher.search(query)
        return results
                
    def filter_documents(self,regex):
        """returns the filtered set of documents matching regex"""
        #TODO should be based on lucene index search
        import re
        # if regex is provided as str compile as regex
        regex = regex if isinstance(regex, re.Pattern) else re.compile(regex,re.IGNORECASE)
        def callback(title,abstract,elem,filteredList):
            if bool(regex.search(title) if title else False) | bool(regex.search(abstract) if abstract else False):
                filteredList.append(
                    {
                    'title': title,
                    'abstract': abstract,
                    'xmlelem': elem
                    }
                )
        filteredList = []
        self.process_documents(callback, filteredList=filteredList)
        return filteredList
    
