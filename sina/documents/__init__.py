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

    def process_documents(self,callback,*args,verbose=False,progress=True,onepass=False,limit=None,**kwargs):
        """process documents with a callback function
        
        Args:
            callback (function): function that will take the title, abstract text, and full
              xml element as first positional arguments and any further provided arguments
            verbose (bool): verbose output
            progress (bool): show progress
            onepass (str): if a str is given, it will track wich documents have already been
              processed with the callback and skip them. It is up to the developer to provide
              a suitable filename that will be used for the tracking.
            limit (int): if limit takes a random sample of the xmlfilenames for testing purposes.
        """
        import hashlib, json
        xmlfilenames = sorted(
            glob.glob(os.path.join(self.location,'*.xml.gz'))
        )
        if limit:
            import random
            xmlfilenames = random.sample(xmlfilenames, limit)
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
            if progress: print(end='\rProgress (%): {:.4f}'.format((xmli+1)/totalxmlfiles)*100)
            if onepass: processedFiles.add(os.path.basename(xmlfilename))
        if onepass:
            json.dump(list(processedFiles), open(os.path.join(self.location, onepass),'wt'))

    def build_document_index(self, shards=10, include_mesh=True, limit=None):
        """Build an index for fast document retrieval with multiprocessing (one process/shard)

        shards (int): The number of document partitionings to use. Whoosh has memory issues
          for large indexes, this is a way around.
        include_mesh (bool): Include mesh terms in shelve db.
        limit (int): If limit, stop indexing after limit number for testing.
        """
        import multiprocessing as mp
        queues = [mp.Queue(maxsize=1000) for i in range(shards)]
        indexdir = os.path.join(self.location,'.index')
        if not os.path.exists(indexdir):
            os.mkdir(indexdir)

        def worker_function(shardnumber):
            from whoosh.index import create_in, open_dir
            from whoosh import fields
            import datetime, sqlite3, shelve, warnings

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

            # shelve <> for time issues avoiding create sqlite db
            dbshelve = shelve.open(os.path.join(indexdir,str(shardnumber),'pmid.shelve'))
            #conn = sqlite3.connect(
            #    os.path.join(indexdir,str(shardnumber),'pmid.db'),
            #    detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES
            #)
            #dbcursor = conn.cursor()
            #if not(dbcursor.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchone()):
            #    # Create table if this is first connection
            #    dbcursor.execute('''CREATE TABLE abstracts
            #        (pmid INTEGER PRIMARY KEY, version INTEGER, date TIMESTAMP, filepos INTEGER, articlepos INTEGER, xmlbytes INTEGER)'''
            #    )
            #    conn.commit()

            import calendar, locale
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            # for available settings see $ locale -a
            # if en_US.UTF-8 this will fail
            months = dict(
                (v,k) for k,v in list(enumerate(calendar.month_abbr))+list(enumerate(calendar.month_name))
            )

            commitCounter = 0
            commitLoop = 1000 #do a commit every 1000 document inserts
            while True:
                title,abstract,elem,position = queues[shardnumber].get()
                if not elem: break

                if elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Year') is not None:
                    # First get month as sometimes it is a number, sometimes 3 letters
                    datemonth = elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Month')
                    if datemonth is not None:
                        datemonth = int(datemonth.text) if datemonth.text.isnumeric() else months[datemonth.text]
                    else: datemonth = 1
                    try: date = datetime.datetime(
                        int(elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Year').text),
                        datemonth,
                        int(elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Day').text) if
                        elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Day') is not None else 1
                        # some PubDate s miss the 'Month' and/or 'Day'
                    )
                    except ValueError as e:
                        # Only catching ValueError here, MedlineCitation dates should be curated by pubmed
                        print(position,e)
                        date = datetime.datetime(1,1,1)
                elif elem.find('MedlineCitation/DateCompleted/Year') is not None:
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
                    date = datetime.datetime(1,1,1) # Dummy date if no date available
                pmidversion = int(elem.find('MedlineCitation/PMID').get('Version'))

                # Extract mesh terms
                if include_mesh:
                    meshterms = [
                        mh.find('DescriptorName').get('UI') #.text for the actual mesh concept
                        for mh in elem.find('MedlineCitation/MeshHeadingList')
                    ] if elem.find('MedlineCitation/MeshHeadingList') is not None else []
                
                # Prepare indexer writer
                if not (commitCounter % commitLoop): writer = ix.writer()
                commitCounter+=1 # this ensures that creating writer and committing occur in subsequent loops
                
                # Check if article has already been indexed or needs updating
                #try:
                #    dbcursor.execute(
                #        'INSERT INTO abstracts(pmid,date,filepos,articlepos) values (?,?,?,?)',
                #        (position[2],date,position[0],position[1])
                #    )
                pmidstr = str(position[2])
                if not pmidstr in dbshelve:
                    dbshelve[pmidstr] = (position[0],position[1],None,date) + ((meshterms,) if include_mesh else ())
                #except sqlite3.IntegrityError:
                #    prevpos = dbcursor.execute(
                #        'SELECT filepos,articlepos,version FROM abstracts WHERE pmid=?',(position[2],)
                #    ).fetchone()
                else:
                    prevpos = dbshelve[pmidstr]
                    if prevpos[0] == position[0]:
                        if prevpos[1] == position[1]:
                            continue # abstract already indexed
                        elif prevpos[2] == pmidversion:
                            warnings.warn('same versioned pmid ({}) in same file ({}) twice'.format(position[2],position[0]))
                    elif prevpos[0] > position[0]:
                        continue # has already been updated in a previous run
                    # should be case where update is required
                    #dbcursor.execute(
                    #    'UPDATE abstracts SET date=?, filepos=?, articlepos=?, version=? WHERE pmid=?',
                    #    # in current setup having a pmidversion of 1 in db indicates at least one update
                    #    (date,position[0],position[1],pmidversion,position[2])
                    #)
                    dbshelve[pmidstr] = (position[0],position[1],pmidversion,date) + ((meshterms,) if include_mesh else ())
                    writer.delete_by_term('pmid', str(position[2]))

                # Indexer code
                writer.add_document(
                    title=title,
                    pmid=elem.find('MedlineCitation/PMID').text,
                    # Including the title in the content so it is also indexed searched
                    content=abstract,#title+(' ' if title.endswith('.') else '. ')+abstract,
                    date=date
                )
                if not (commitCounter % commitLoop):
                    writer.commit()
                    #conn.commit()
            if (commitCounter % commitLoop):
                # last commit if not committed in last loop
                writer.commit()
                #conn.commit()
            ix.close()
            #conn.close()
            dbshelve.close()

        def commit_abstracts(title,abstract,elem,position):
            queues[position[2] % shards].put([title,abstract,elem,position])
            #position[2] == pmid

        # Start up workers
        processes = [mp.Process(target=worker_function,args=(i,)) for i in range(shards)]
        for p in processes: p.start()
        # Build index
        self.process_documents(commit_abstracts, onepass='.indexed_files.json', limit=limit)
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
            print('\rProcessing shard %s/%s' % (os.path.basename(indexdir),total_idirs-1), end='')
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
        xmlfilenames = sorted(
            glob.glob(os.path.join(self.location,'*.xml.gz'))
        )
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

class PubmedCentralCollection(BaseDocumentCollection):
    """Example:

    >>> pmc = PubmedCentralCollection('pubmedcentral','~/pubmedcentral')
    >>> pmc.retrieve_documents()
    """
    def retrieve_documents(self):
        import ftplib
        from sina.config import secrets
        ftplocations = [
            '/pub/pmc/oa_bulk', # open access commercial and non-commercial use
            # PMID mapping file seems to be PMC-ids.csv.gz in /pub/pmc
            '/pub/pmc/manuscript', # author manuscript collection
            '/pub/pmc/historical_ocr', # historical ocr collection
        ]
        ftp = ftplib.FTP(
            host='ftp.ncbi.nlm.nih.gov',
            user='anonymous',
            passwd=secrets.getsecret('email')
        )
        for ftplocation in ftplocations:
            ftp.cwd(ftplocation)
            filenames = ftp.nlst()
            collection = os.path.basename(ftplocation)
            if not os.path.exists(os.path.join(self.location,collection)):
                os.mkdir(os.path.join(self.location,collection))
            for filename in filenames:
                if filename.endswith('.tmp'):
                    continue # skip if retrieving while ncbi is making update
                localfilename = os.path.join(self.location,collection,filename)
                # no md5 s at locations for pmc
                if not os.path.exists(localfilename):
                    print('Retrieving',filename)
                    with open(localfilename,'wb') as fh:
                        ftp.retrbinary('RETR '+ filename, fh.write)
        ftp.close() #ftp.quit()

    def process_documents(self):
        raise NotImplementedError
    
    def filter_documents(self):
        raise NotImplementedError

class PubmedQueryResult(object):
    """Result from a Pubmed index query
    Methods for processing the result

    Args:
      corpus (PubmedCollection): Pubmed corpus that provided results.
      results (list): Document result set.
      test_fraction (float): If provided a random subset equaling test_fraction
        is set apart for testing purposes after machine learning tasks downstream.
    """
    def __init__(self, corpus, results, test_fraction=.25):
        import pandas as pd
        self.results = pd.DataFrame(results).set_index('pmid')
        self.corpus = corpus
        if test_fraction:
            from sklearn.model_selection import train_test_split
            self.results, self.results_test = train_test_split(self.results, test_size=test_fraction)
            self.testset = True
        else: self.testset = False

    def analyze_mesh(self,topfreqs=20,getmeshnames=False):
        # wget ftp://nlmpubs.nlm.nih.gov/online/mesh/MESH_FILES/xmlmesh/desc2019.gz
        from collections import Counter
        import shelve, pandas as pd
        shards = sorted(glob.glob(os.path.join(self.corpus.location,'.index/*')))
        lenshards = len(shards)
        pmiddbs = [
            shelve.open(os.path.join(self.corpus.location,'.index',str(i),'pmid.shelve')) for i in range(lenshards)
        ]
        self.meshterms = [pmiddbs[int(pmid)%lenshards][pmid][4] for pmid in self.results.index]
        if self.testset:
            self.meshterms_test = [pmiddbs[int(pmid)%lenshards][pmid][4] for pmid in self.results_test.index]
        self.meshfreqs = Counter((mt for a in self.meshterms for mt in a))

        # Filter top
        self.meshtop = pd.DataFrame(
            self.meshfreqs.most_common(topfreqs),
            columns=['meshid','freq']
        )
        self.topfreqs = topfreqs
        if getmeshnames:
            import gzip, xml.etree.ElementTree as ET
            with gzip.open('/home/mint/LSData/private/desc2019.gz') as xmlfilehandle:
                xml = ET.parse(xmlfilehandle)
                root = xml.getroot()
                self.meshtop['meshname'] = self.meshtop.meshid.apply(
                    lambda x: root.find(
                        "DescriptorRecord[DescriptorUI='{}']/DescriptorName/String".format(x)
                        ).text
                )
                del xml, root
        
        meshtop_ix = pd.Index(self.meshtop.meshid)
        self.meshtabel = pd.DataFrame(
            {pmid:meshtop_ix.isin(self.meshterms[i]) for i,pmid in enumerate(self.results.index)}
        ).T
        self.meshtabel['mtindexsum'] = self.meshtabel.sum(axis=1)
        self.meshtabel.sort_values('mtindexsum',ascending=False,inplace=True)
        del self.meshtabel['mtindexsum']
        if self.testset:
            self.meshtabel_test = pd.DataFrame(
                {pmid:meshtop_ix.isin(self.meshterms_test[i]) for i,pmid in enumerate(self.results_test.index)}
            ).T

    def predict_meshterms(self, only_freqs=False):
        import numpy as np
        from sklearn import svm, naive_bayes, metrics
        #from gensim.corpora import Dictionary
        #from gensim.models import TfidfModel
        # Transform text
        #embedding = self.gensim_w2v(doclevel=True)
        #dct = Dictionary(self.processed_documents)
        #dct.id2token = {v: k for k, v in dct.token2id.items()}
        #corpus = [dct.doc2bow(line) for line in self.processed_documents]
        #model = TfidfModel(corpus)
        #X = [
        #    [
        #        (i,np.append(
        #            embedding.wv[dct.id2token[i]] if dct.id2token[i] in embedding.wv else np.zeros(embedding.wv.vector_size),
        #            f
        #        )) for i,f in model[doc]
        #    ]
        #    for doc in corpus
        #]
        # Prepare data
        X = np.vstack(
            [self.normalize_text_length(doc, only_freqs) for doc in self.processed_documents]
        )
        X_test = np.vstack(
            [self.normalize_text_length(doc, only_freqs) for doc in self.processed_documents_test]
        )
        Y = self.meshtabel.values
        Y_test = self.meshtabel_test.values

        # ML model
        self.models = []
        accuracies = []
        precisions = []
        recalls = []
        for i in range(Y.shape[1]):
            #model = naive_bayes.GaussianNB()
            model = svm.SVC(kernel='rbf', class_weight='balanced', gamma='scale')
            model.fit(X, Y[:,i])
            Y_test_pred = model.predict(X_test)
            accuracies.append(metrics.accuracy_score(Y_test[:,i], Y_test_pred))
            precisions.append(metrics.precision_score(Y_test[:,i], Y_test_pred))
            recalls.append(metrics.recall_score(Y_test[:,i], Y_test_pred))
            print(i, accuracies[-1])
            self.models.append(model)
        self.meshtop['pred_acc'] = accuracies
        self.meshtop['pred_prec'] = precisions
        self.meshtop['pred_rec'] = recalls
        self.X, self.X_test, self.Y, self.Y_test = X, X_test, Y, Y_test
        
    def sentence_embedding(self, test_size=.2, random_state=None):
        """WORK IN PROGRESS"""
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from keras.models import Sequential
        from keras.layers import Dense
        
        # Align meshtabel and results index as to place
        # meshtabel value lists in results dataframe
        self.meshtabel.sort_index(inplace=True)
        self.results.sort_index(inplace=True)
        self.results['meshterms'] = list(self.meshtabel.values)
        self.results['meshsum'] = self.results.meshterms.apply(sum)
        self.results['domainspec'] = self.results.meshsum > self.results.meshsum.median()
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            self.results.title, self.results.domainspec,
            test_size=test_size, random_state=random_state
        )
        
        model = Sequential()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        
    def gensim_topics(self):
        import gensim
        import spacy
        nlp = spacy.load('en')
        #nlp = spacy.load('en', disable=['ner', 'parser']) # disabling NER and POS tagging for speed
        corpus = [
            [token.lemma_.lower() for token in nlp(c) if not (token.is_stop or token.is_punct)]
            for c in self.results.content
        ]
        dictionary = gensim.corpora.Dictionary(corpus)
        corpus_bow = [dictionary.doc2bow(c) for c in corpus]
        ldamodel = gensim.models.LdaModel(
            corpus=corpus_bow, num_topics=10, id2word=dictionary, iterations=100
        )
        return ldamodel

    def gensim_w2v(self, vecsize=100, bigrams=True, split_hyphens=False, doclevel=False):
        """Build word2vec model with gensim
        
        Args:
          vecsize (int): Size of the embedding vectors
          bigrams (bool|int): if True or 1 build bigram, if 2 build trigram, etc
          split_hyphens (bool): split hyphened words into subtokens
          doclevel (bool): also process document level with sentence processing
        """
        import spacy, gensim
        nlp = spacy.load('en', disable=['ner', 'parser'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        if not split_hyphens:
            # Prevent splitting intra-word hyphens
            suffixes = nlp.Defaults.suffixes + (r'''\w+-\w+''',)
            suffix_regex = spacy.util.compile_suffix_regex(suffixes)
            nlp.tokenizer.suffix_search = suffix_regex.search
        sentences = [
            [
            token.lemma_.lower() for token in sent
            if not (token.is_stop or token.is_punct or token.is_digit)
            ]
            for txt in self.results.content for sent in nlp(txt).sents
        ]
        if doclevel:
            self.processed_documents = [
                [
                token.lemma_.lower() for sent in nlp(txt).sents for token in sent
                if not (token.is_stop or token.is_punct or token.is_digit)
                ]
                for txt in self.results.content
            ]
            if self.testset:
                self.processed_documents_test = [
                    [
                    token.lemma_.lower() for sent in nlp(txt).sents for token in sent
                    if not (token.is_stop or token.is_punct or token.is_digit)
                    ]
                    for txt in self.results_test.content
                ]
        while bigrams:
            bigrams-=1 # for trigram set bigrams=2
            bigram = gensim.models.Phrases(
                sentences, min_count=30, progress_per=10000,
                delimiter=chr(183).encode() # interpunct aka &middot; as delimiter
            )
            sentences = [bigram[sent] for sent in sentences]
            if doclevel:
                self.processed_documents = [bigram[doc] for doc in self.processed_documents]
                if self.testset:
                    self.processed_documents_test = [bigram[doc] for doc in self.processed_documents_test]
        # todo put this code for processing sentences in function so it can also
        # be reused for classification
        self.processed_sentences = sentences
        w2model = gensim.models.word2vec.Word2Vec(sentences, size=vecsize, hs=1)
        
        # visualize
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        # Get 100 most frequent words
        top100words = w2model.wv.index2entity[:100]
        top100vectors = w2model.wv[top100words]

        # Reduce with PCA dimensionality as tSNE does not scale well
        top100vectors_reduc = PCA(n_components=50).fit_transform(top100vectors) if vecsize > 50 else top100vectors
        Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(top100vectors_reduc)

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(Y[:,0], Y[:,1])
        for word, vec in zip(top100words, Y):
            ax.text(vec[0], vec[1],
                 word.title(),
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 size='medium',
            ).set_size(15)

        self.embedding = w2model
        return w2model

    def k_means_embedding(self, k=100, verbose=False):
        """Calculate k means centroids
        for the word embedding space

        Args:
          k (int): Number of centroids, defaults to 100.
          verbose (bool): Print closest word vectors to calculated centroids.
        """
        #from sklearn.neighbors import KNeighborsClassifier
        from sklearn.cluster import KMeans
        kmeans = KMeans(k, init='k-means++')
        kmeans.fit(self.embedding.wv[self.embedding.wv.index2entity])
        if verbose:
            for i,center in enumerate(kmeans.cluster_centers_):
                print(self.embedding.wv.similar_by_vector(center)[0])
        self.embedding_kmeans = kmeans

    def normalize_text_length(self, textlist, only_freqs=False):
        """Using k centroids generated by self.k_means_embedding
        transform text to average vectors with vector count per cluster

        Args:
          textlist (list): A list of processed tokens representing the text.
            Should be processed in the same manner as the embedding.
          only_freqs (bool): Do not include wordvectors.
        """
        import pandas as pd, numpy as np
        textvectors = [
            self.embedding.wv[token] for token in textlist if token in self.embedding.wv
        ]
        predictions = pd.DataFrame({
            'label': self.embedding_kmeans.predict(textvectors),
            'wordvector': textvectors
        })
        clustercounts = predictions.groupby('label').count()
        # normalize for length of abstract
        if clustercounts.wordvector.sum():
            clustercounts.wordvector = clustercounts.wordvector/clustercounts.wordvector.sum()
        clustercounts['avgvec'] = predictions.groupby('label').apply(
            lambda x: np.vstack(x.wordvector).mean(axis=0)
        )
        emptyclusters = pd.DataFrame({
            i:{'wordvector':0,'avgvec':np.zeros(self.embedding_kmeans.cluster_centers_.shape[1])}
            for i in range(self.embedding_kmeans.cluster_centers_.shape[0]) if i not in clustercounts.index
        }).T
        clustercounts = pd.concat((clustercounts, emptyclusters), sort=True).sort_index()
        if only_freqs:
            return clustercounts.wordvector.values
        else:
            # Returning one large array with first cluster counts and then
            # all appended average vectors of non-empty clusters
            return np.append(
                clustercounts.wordvector.values,
                np.hstack(clustercounts.avgvec)
            )

    @staticmethod
    def embedding_projection(emb1, emb2, topvoc_emb2=None, test_similarity=False, custom=True):
        """Get embedding projection from one embedding to another
        
        Args:
          emb1 (gensim.models.word2vec.Word2Vec): Reference embedding
          emb2 (gensim.models.word2vec.Word2Vec): Target embedding
          topvoc_emb2 (int): If given restrict reference words to topwords
            of target embedding
          test_similarity (bool): Test the cosine similarity
          custom (bool): Custom numpy algorithm for projection

        Returns: 
          projection function, that takes a word from emb1 and returns
          its vector in emb2

        Example:
        >>> pmc = PubmedCollection('pubmed','~/pubmed'
        >>> emb1 = PubmedQueryResult(
        ...   results=pmc.query_document_index('neuroblastoma OR cancer'),corpus=pmc
        ... ).gensim_w2v()
        >>> emb2 = PubmedQueryResult(
        ...   results=pmc.query_document_index('neuroblastoma'),corpus=pmc
        ... ).gensim_w2v()
        >>> PubmedQueryResult.embedding_projection(emb1, emb2)
        """
        import gensim, numpy as np, random
        emb2vocab = list(emb2.wv.vocab)[:topvoc_emb2] if topvoc_emb2 else emb2.wv.vocab

        # Allows projecting known words from emb1 into emb2 even if unkown
        if custom:
            commonwords = list(set(emb1.wv.vocab)&set(emb2vocab))
            sourcevec = emb1.wv[commonwords]
            targetvec = emb2.wv[commonwords]
            n, dim = targetvec.shape
            centeredS = sourcevec - sourcevec.mean(axis=0)
            centeredT = targetvec - targetvec.mean(axis=0)
            C = np.dot(np.transpose(centeredS), centeredT) / n
            V,S,W = np.linalg.svd(C)
            d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
            if d:
                S[-1] = -S[-1]
                V[:, -1] = -V[:, -1]
            R = np.dot(V, W)
            varS = np.var(sourcevec, axis=0).sum()
            c = 1/varS * np.sum(S) # scale factor
            t = targetvec.mean(axis=0) - sourcevec.mean(axis=0).dot(c*R)
            #np.allclose(sourcevec.dot(c*R) + t, targetvec)
            err = ((sourcevec.dot(c * R) + t - targetvec) ** 2).sum()
            print('Error on training points:', err)
            projfunc = lambda x: emb1.wv[x].dot(c * R) + t
        else:
            commonwords = [(w,w) for w in set(emb1.wv.vocab)&set(emb2vocab)]
            transmat = gensim.models.translation_matrix.TranslationMatrix(emb1.wv, emb2.wv)
            transmat.train(commonwords)
            # emb2.wv.similar_by_vector(np.dot(transmat.translation_matrix,emb1.wv['cyclin-dependent']),topn=3)
            # ==
            # transmat.translate(['cyclin-dependent'], topn=3)
            projfunc = lambda x: np.dot(transmat.translation_matrix,emb1.wv[x])

        if test_similarity and topvoc_emb2:
            # For the moment only sensible to test if restricting reference points emb2
            from scipy import spatial
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            controlset = random.sample(list(emb2.wv.vocab)[topvoc_emb2:],topvoc_emb2)
            trained_cosines = [
                spatial.distance.cosine(projfunc(w), emb2.wv[w])
                for w in emb2vocab if w in emb1.wv.vocab
            ]
            control_cosines = [
                spatial.distance.cosine(projfunc(w), emb2.wv[w])
                for w in controlset if w in emb1.wv.vocab
            ]
            print(
                'Avg+std trained cosine sim:', np.mean(trained_cosines), np.std(trained_cosines),
                ', avg+std control cosine sim:', np.mean(control_cosines), np.std(control_cosines)
            )

            # Plotting
            pca = PCA(n_components=2).fit(emb2.wv[emb2.wv.index2entity])
            fig, ax = plt.subplots()
            ## emb training set
            Y = pca.transform(emb2.wv[emb2vocab])
            Y_projection = pca.transform(
                np.stack([projfunc(w) for w in emb2vocab if w in emb1.wv.vocab])
            )
            ax.scatter(Y[:,0], Y[:,1], label='original_trained')
            ax.scatter(Y_projection[:,0], Y_projection[:,1], label='projection_trained')
            for word, vec in list(zip(emb2vocab, Y))+list(zip(emb2vocab, Y_projection)):
                ax.text(vec[0], vec[1],
                    word.title(),
                ).set_size(15)
            ## control set
            Y_control = pca.transform(emb2.wv[controlset])
            Y_control_projection = pca.transform(
                np.stack([projfunc(w) for w in controlset if w in emb1.wv.vocab])
            )
            ax.scatter(Y_control[:,0], Y_control[:,1], label='original_control')
            ax.scatter(Y_control_projection[:,0], Y_control_projection[:,1], label='projection_control')
            for word, vec in list(zip(controlset, Y_control))+list(zip(controlset, Y_control_projection)):
                ax.text(vec[0], vec[1],
                    word.title(),
                ).set_size(15)
            ## legend
            ax.legend()
            
        return projfunc
