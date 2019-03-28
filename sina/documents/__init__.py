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
    def retrieve_documents(self,check_md5=False):
        import ftplib, hashlib
        ftp = ftplib.FTP(
            host='ftp.ncbi.nlm.nih.gov',
            user='anonymous',
            passwd='christophe.vanneste@kaust.edu.sa' #TODO fetch from kindi
        )
        ftp.cwd('pubmed/baseline')
        filenames = ftp.nlst()
        for filename in filenames:
            localfilename = os.path.join(self.location,filename)
            # md5 logic
            if check_md5 and localfilename.endswith('.xml.gz'):
                try:
                    md5 = open(localfilename+'.md5').read().strip().split()[1]
                except FileNotFoundError: refetch = False
                try:
                    with open(localfilename,'rb') as file_to_check:
                        md5_returned = hashlib.md5(file_to_check.read()).hexdigest()
                        if md5 != md5_returned:
                            refetch = True
                        else: refetch = False
                except FileNotFoundError:
                    refetch = True
            else: refetch = False
            if refetch or not os.path.exists(localfilename):
                print('Retrieving',filename)
                with open(localfilename,'wb') as fh:
                    ftp.retrbinary('RETR '+ filename, fh.write)
        ftp.quit()

    def process_documents(self,callback,*args,verbose=False,progress=True,**kwargs):
        """process documents with a callback function
        
        Args:
            callback (function): function that will take the title, abstract text, and full
              xml element as first positional arguments and any further provided arguments
        """
        import hashlib
        xmlfilenames = glob.glob(os.path.join(self.location,'*.xml.gz'))
        totalxmlfiles = len(xmlfilenames)
        article_pos = 0
        for xmli,xmlfilename in enumerate(xmlfilenames):
            #check md5
            md5 = open(xmlfilename+'.md5').read().strip().split()[1]
            with open(xmlfilename,'rb') as file_to_check:
                md5_returned = hashlib.md5(file_to_check.read()).hexdigest()
                if md5 != md5_returned:
                    print('Run the method `pmc.retrieve_documents(check_md5=True)` on the pubmed object')
                    raise FileNotFoundError('Correct file was not downloaded',xmlfilename)
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
                            callback(title,abstract,elem,(xmli,article_pos),*args,**kwargs) #TODO save map xmli to filename
                        except AttributeError as e:
                            if verbose: print(e)
                        article_pos += 1
                        #TODO output article
                        root.clear() # ensures only one article is kept in memory
                        # but because of the 'clear' you cannot use the root for info on
                        # current iterated PubmedArticle element
            if progress: print(end='\rProgress (%): {:<10}'.format(xmli/totalxmlfiles)*100)

    def build_document_index(self, shards=10):
        """Build an index for fast document retrieval

        shards (int): The number of document partitionings to use. Whoosh has memory issues
          for large indexes, this is a way around.
        """
        from whoosh.index import create_in
        from whoosh import fields
        schema = fields.Schema(
            title=fields.TEXT(stored=True),
            pmid=fields.ID(stored=True),
            content=fields.TEXT(stored=True),
            date=fields.DATETIME(stored=True),
            #filepos=fields.INT(),
            #articlepos=fields.INT()
        )
        indexdir = os.path.join(self.location,'.index')
        os.mkdir(indexdir)
        ix = []
        for iix in range(shards):
            os.mkdir(os.path.join(indexdir,str(iix)))
            ix.append(create_in(os.path.join(indexdir,str(iix)), schema))
        def commit_abstracts(title,abstract,elem,position):
            import datetime
            date = datetime.datetime(
                int(elem.find('MedlineCitation/DateCompleted/Year').text),
                int(elem.find('MedlineCitation/DateCompleted/Month').text),
                int(elem.find('MedlineCitation/DateCompleted/Day').text)
            )
            writer = ix[position[0] % 10].writer() #make this position[0]*position[1] to redistribute
            writer.add_document(
                title=title,
                pmid=elem.find('MedlineCitation/PMID').text,
                content=abstract,
                date=date
            )
            writer.commit()
        self.process_documents(commit_abstracts)

    def query_document_index(self,query,sortbydate=False):
        """Query the corpus index

        Args:
            query (str): the elasticv-search like query str to run.
            sortbydate (bool): If set to True, might encounter memory
              issues with the current version of whoosh.
        """
        import whoosh.index as index
        from whoosh.qparser import QueryParser
        indexdirs = os.path.join(self.location,'.index','*')
        total_idirs = len(glob.glob(indexdirs))
        results = []
        querystr = query
        print()
        for indexdir in range(total_idirs):
            indexdir = os.path.join(self.location,'.index',str(indexdir))
            print('\rProcessing shard %s/%s' % (os.path.basename(indexdir),total_idirs), end='')
            ix = index.open_dir(indexdir)
            query = QueryParser("content", ix.schema).parse(querystr)
            with ix.searcher() as searcher:
                search_results = searcher.search(query, limit=None, scored=False, sortedby='date' if sortbydate else None)
                results += [r.fields() for r in search_results]
        print('\n',len(results),'retrieved')
        return results
                
    def filter_documents(self,regex):
        """returns the filtered set of documents matching regex"""
        #TODO should be based on lucene index search
        import re
        # if regex is provided as str compile as regex
        regex = regex if isinstance(regex, re.Pattern) else re.compile(regex,re.IGNORECASE)
        def callback(title,abstract,elem,position,filteredList):
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
    
    def build_xml_index(self):
        """Iterates over all the documents and saves for each pmid the file and corresponding
        positions.
        """
        import sqlite3, re
        xmlfilenames = glob.glob(os.path.join(self.location,'*.xml.gz'))
        articlere = re.compile(b'<PubmedArticle>.+?</PubmedArticle>', re.MULTILINE | re.DOTALL)
        pmidre = re.compile(b'<PMID.+?>(\d+)</PMID>')
        conn = sqlite3.connect(os.path.join(self.location,'pmid_locations.db'))
        c = conn.cursor()
        c.execute('''CREATE TABLE ablocations
             (pmid text, filename text, start integer, length integer)''')
        conn.commit()
        for i,xmlfilename in enumerate(xmlfilenames):
            print('\r',i,end='',sep='')
            basename = os.path.basename(xmlfilename)
            with gzip.open(xmlfilename) as xmlfilehandler:
                xmlcontent = xmlfilehandler.read()
                for article in articlere.finditer(xmlcontent):
                    pmid = pmidre.search(article.group()).groups()[0].decode()
                    c.execute("INSERT INTO ablocations VALUES ('{}','{}',{},{})".format(
                        pmid, basename, article.start(), article.end()-article.start()
                    ))
            conn.commit()
        conn.close()
