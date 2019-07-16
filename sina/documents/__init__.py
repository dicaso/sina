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
    def retrieve_documents(self,ftplocation='pubmed/baseline',check_md5=False):
        import ftplib, hashlib
        from sina.config import secrets
        ftp = ftplib.FTP(
            host='ftp.ncbi.nlm.nih.gov',
            user='anonymous',
            passwd=secrets.getsecret('email')
        )
        ftp.cwd(ftplocation)
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
        ftp.close() #ftp.quit()

    def update(self):
        """Retrieve new documents and process them.
        Could be used to start from scratch.

        Updated pubmed xml documents contain new, revised and deleted abstracts,
        and should be handled accordingly.
        """
        self.retrieve_documents(check_md5=True)
        self.retrieve_documents(ftplocation='pubmed/updatefiles',check_md5=True)

    def process_documents(self,callback,*args,verbose=False,progress=True,onepass=False,**kwargs):
        """process documents with a callback function
        
        Args:
            callback (function): function that will take the title, abstract text, and full
              xml element as first positional arguments and any further provided arguments
            verbose (bool): verbose output
            progress (bool): show progress
            onepass (str): if a str is given, it will track wich documents have already been
              processed with the callback and skip them. It is up to the developer to provide
              a suitable filename that will be used for the tracking.
        """
        import hashlib, json
        xmlfilenames = glob.glob(os.path.join(self.location,'*.xml.gz'))
        totalxmlfiles = len(xmlfilenames)
        article_pos = 0
        if onepass:
            try: processedFiles = set(json.load(open(os.path.join(self.location, onepass))))
            except FileNotFoundError: processedFiles = set()
        for xmli,xmlfilename in enumerate(xmlfilenames):
            if onepass and os.path.basename(xmlfilename) in processedFiles: continue
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
                            pmid = int(elem.find('MedlineCitation/PMID').text)
                            title = elem.find('MedlineCitation/Article/ArticleTitle').text
                            abstract = elem.find('MedlineCitation/Article/Abstract/AbstractText').text
                            callback(title,abstract,elem,(xmli,article_pos,pmid),*args,**kwargs) #TODO save map xmli to filename
                        except AttributeError as e:
                            if verbose: print(e)
                        article_pos += 1
                        #TODO output article
                        root.clear() # ensures only one article is kept in memory
                        # but because of the 'clear' you cannot use the root for info on
                        # current iterated PubmedArticle element
            if progress: print(end='\rProgress (%): {:.4f}'.format(xmli/totalxmlfiles)*100)
            if onepass: processedFiles.add(os.path.basename(xmlfilename))
        if onepass:
            json.dump(list(processedFiles), open(os.path.join(self.location, onepass),'wt'))

    def build_document_index(self, shards=10):
        """Build an index for fast document retrieval

        shards (int): The number of document partitionings to use. Whoosh has memory issues
          for large indexes, this is a way around.
        """
        from whoosh.index import create_in, open_dir
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
        if not os.path.exists(indexdir):
            os.mkdir(indexdir)
        ix = []
        for iix in range(shards):
            if not os.path.exists(os.path.join(indexdir,str(iix))):
                os.mkdir(os.path.join(indexdir,str(iix)))
                ix.append(create_in(os.path.join(indexdir,str(iix)), schema))
            else: ix.append(open_dir(os.path.join(indexdir,str(iix)), schema=schema))
        def commit_abstracts(title,abstract,elem,position):
            import datetime
            if elem.find('MedlineCitation/DateCompleted/Year') is not None:
                date = datetime.datetime(
                    int(elem.find('MedlineCitation/DateCompleted/Year').text),
                    int(elem.find('MedlineCitation/DateCompleted/Month').text),
                    int(elem.find('MedlineCitation/DateCompleted/Day').text)
                )
            else:
                date = datetime.datetime(
                    int(elem.find('MedlineCitation/DateRevised/Year').text),
                    int(elem.find('MedlineCitation/DateRevised/Month').text),
                    int(elem.find('MedlineCitation/DateRevised/Day').text)
                )
            writer = ix[position[2] % 10].writer() #make this position[0]*position[1] to redistribute
            writer.add_document(
                title=title,
                pmid=elem.find('MedlineCitation/PMID').text,
                content=abstract,
                date=date
            )
            writer.commit()
        self.process_documents(commit_abstracts, onepass='.indexed_files.json')

    def build_document_index_mp(self, shards=10):
        """Build an index for fast document retrieval with multiprocessing (one process/shard)

        shards (int): The number of document partitionings to use. Whoosh has memory issues
          for large indexes, this is a way around.
        """
        import multiprocessing as mp
        queues = [mp.Queue(maxsize=1000) for i in range(shards)]
        indexdir = os.path.join(self.location,'.index')
        if not os.path.exists(indexdir):
            os.mkdir(indexdir)

        def worker_function(shardnumber):
            from whoosh.index import create_in, open_dir
            from whoosh import fields
            import datetime, sqlite3, warnings

            schema = fields.Schema(
                title=fields.TEXT(stored=True),
                pmid=fields.ID(stored=True,unique=True),
                content=fields.TEXT(stored=True),
                date=fields.DATETIME(stored=True),
                #filepos=fields.INT(),
                #articlepos=fields.INT()
            )

            if not os.path.exists(os.path.join(indexdir,str(shardnumber))):
                os.mkdir(os.path.join(indexdir,str(shardnumber)))
                ix = create_in(os.path.join(indexdir,str(shardnumber)), schema)
            else:
                ix = open_dir(os.path.join(indexdir,str(shardnumber)), schema=schema)

            # create sqlite db
            conn = sqlite3.connect(
                os.path.join(indexdir,str(shardnumber),'pmid.db'),
                detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES
            )
            dbcursor = conn.cursor()
            if not(dbcursor.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchone()):
                # Create table if this is first connection
                dbcursor.execute('''CREATE TABLE abstracts
                    (pmid INTEGER PRIMARY KEY, version INTEGER, date TIMESTAMP, filepos INTEGER, articlepos INTEGER, xmlbytes INTEGER)'''
                )
                conn.commit()

            commitCounter = 0
            commitLoop = 1000 #do a commit every 1000 document inserts
            while True:
                title,abstract,elem,position = queues[shardnumber].get()
                if not elem: break

                if elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Year') is not None:
                    date = datetime.datetime(
                        int(elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Year').text),
                        int(elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Month').text) if,
                        elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Month') is not None else 1
                        int(elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Day').text) if
                        elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Day') is not None else 1
                        # some PubDate s miss the 'Month' and/or 'Day'
                    )
                if elem.find('MedlineCitation/DateCompleted/Year') is not None:
                    date = datetime.datetime(
                        int(elem.find('MedlineCitation/DateCompleted/Year').text),
                        int(elem.find('MedlineCitation/DateCompleted/Month').text),
                        int(elem.find('MedlineCitation/DateCompleted/Day').text)
                    )
                elif elem.find('MedlineCitation/DateRevised/Year') is not None:
                    date = datetime.datetime(
                        int(elem.find('MedlineCitation/DateRevised/Year').text),
                        int(elem.find('MedlineCitation/DateRevised/Month').text),
                        int(elem.find('MedlineCitation/DateRevised/Day').text)
                    )
                else:
                    print('No date for',position)
                    continue #skipping entries without date
                pmidversion = int(elem.find('MedlineCitation/PMID').get('Version'))
                
                # Prepare indexer writer
                if not (commitCounter % commitLoop): writer = ix.writer()
                commitCounter+=1 # this ensures that creating writer and committing occur in subsequent loops
                
                # Check if article has already been indexed or needs updating
                try:
                    dbcursor.execute(
                        'INSERT INTO abstracts(pmid,date,filepos,articlepos) values (?,?,?,?)',
                        (position[2],date,position[0],position[1])
                    )
                except sqlite3.IntegrityError:
                    prevpos = dbcursor.execute(
                        'SELECT filepos,articlepos,version FROM abstracts WHERE pmid=?',(position[2],)
                    ).fetchone()
                    if prevpos[0] == position[0]:
                        if prevpos[1] == position[1]:
                            continue # abstract already indexed
                        elif prevpos[2] == pmidversion:
                            warnings.warn('same versioned pmid ({}) in same file ({}) twice'.format(position[2],position[0]))
                    elif prevpos[0] > position[0]:
                        continue # has already been updated in a previous run
                    # should be case where update is required
                    dbcursor.execute(
                        'UPDATE abstracts SET date=?, filepos=?, articlepos=?, version=? WHERE pmid=?',
                        # in current setup having a pmidversion of 1 in db indicates at least one update
                        (date,position[0],position[1],pmidversion,position[2])
                    )
                    writer.delete_by_term('pmid', str(position[2]))

                # Indexer code
                writer.add_document(
                    title=title,
                    pmid=elem.find('MedlineCitation/PMID').text,
                    # Including the title in the content so it is also indexed searched
                    content=title+(' ' if title.endswith('.') else '. ')+abstract,
                    date=date
                )
                if not (commitCounter % commitLoop):
                    writer.commit()
                    conn.commit()
            if (commitCounter % commitLoop):
                # last commit if not committed in last loop
                writer.commit()
                conn.commit()
            ix.close()
            conn.close()

        def commit_abstracts(title,abstract,elem,position):
            queues[position[2] % shards].put([title,abstract,elem,position])
            #position[2] == pmid

        # Start up workers
        processes = [mp.Process(target=worker_function,args=(i,)) for i in range(shards)]
        for p in processes: p.start()
        # Build index
        self.process_documents(commit_abstracts, onepass='.indexed_files.json')
        # Wait for all workers to finish
        for q in queues: q.put([None,None,None,None]) # putting stop signal in each queue
        for p in processes: p.join()

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

        The last entry of an article will overwrite the earlier version.
        """
        import re, shelve#, sqlite3
        xmlfilenames = glob.glob(os.path.join(self.location,'*.xml.gz'))
        articlere = re.compile(b'<PubmedArticle>.+?</PubmedArticle>', re.MULTILINE | re.DOTALL)
        pmidre = re.compile(b'<PMID.+?>(\d+)</PMID>')
        locshelve = shelve.open(os.path.join(self.location,'pmid_locations.shelve'))
        # Chose a shelve instead of sqlite db for time reasons
        #  In: %timeit locshelve[pmid]
        #  2.96 µs ± 462 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        #  In: %timeit c.execute('SELECT * FROM ablocations WHERE pmid=?', (pmid,)).fetchone()
        #  5.72 s ± 38.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        #conn = sqlite3.connect(os.path.join(self.location,'pmid_locations.db'))
        #c = conn.cursor()
        #c.execute('''CREATE TABLE ablocations
        #     (pmid text, filename text, start integer, length integer)''')
        #conn.commit()
        for i,xmlfilename in enumerate(xmlfilenames):
            print('\r',i,end='',sep='')
            basename = os.path.basename(xmlfilename)
            with gzip.open(xmlfilename) as xmlfilehandler:
                xmlcontent = xmlfilehandler.read()
                for article in articlere.finditer(xmlcontent):
                    pmid = pmidre.search(article.group()).groups()[0].decode()
                    locshelve[pmid] = (basename, article.start(), article.end()-article.start())
                    #c.execute(
                    #    "INSERT INTO ablocations VALUES ('?','?',?,?)",
                    #    (pmid, basename, article.start(), article.end()-article.start())
                    #)
            #conn.commit()
        #conn.close()
        locshelve.close()

    def retrieve_article_xmls(self, pmids):
        """Retrieve the xml of a set of articles

        Args:
            pmids (list): List or any iterable of pmid ids in str format

        Returns:
            dict of xml trees: pmid key and xml tree value for each article requested
        """
        import shelve #sqlite3
        locshelve = shelve.open(os.path.join(self.location,'pmid_locations.shelve'))
        #conn = sqlite3.connect(os.path.join(self.location,'pmid_locations.db'))
        articles = {}
        #c = conn.cursor()
        for pmid in pmids:
            #c.execute('SELECT * FROM ablocations WHERE pmid=?', (pmid,))
            abref = (pmid,)+ locshelve[pmid] #c.fetchone()
            with gzip.open(os.path.join(self.location,abref[1])) as xmlfilehandler:
                xmlfilehandler.seek(abref[2])
                xmlcontent = xmlfilehandler.read(abref[3])
                articles[pmid] = ET.fromstring(xmlcontent)
        locshelve.close()
        #conn.close()
        return articles
