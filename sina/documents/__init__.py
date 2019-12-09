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
import sina.config
import logging
import os
import glob
import gzip
import xml.etree.ElementTree as ET


class BaseDocumentCollection(ABC):
    """Base class for document collections

    Args:
        name (str): Name for the collection. If not provided will be class name
          in lowercase with 'collection' removed from the name.
        location (str): Where to store in filesystem. If not provided will be
          stored under `sina.config.appdirs.user_data_dir`/`name` subfolder.
        remote (str): Remote location.
        startTime, endTime, timeUnit -> not yet implemented but should be provided
          for selecting time segments of documents.
    """

    def __init__(self, name=None, location=None, remote=None,
                 startTime=None, endTime=None, timeUnit=1):
        self.name = name or self.__class__.__name__.lower().replace('collection', '')
        self.location = os.path.expanduser(location) if location else os.path.join(
            sina.config.appdirs.user_data_dir, self.name
        )
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
# corpus class


class PubmedCollection(BaseDocumentCollection):
    """Example:

    >>> pmc = PubmedCollection('pubmed')
    >>> pmc.retrieve_documents()
    """

    def retrieve_documents(self, ftplocation='pubmed/baseline', check_md5=False):
        import ftplib
        import hashlib
        from sina.config import secrets
        ftp = ftplib.FTP(
            host='ftp.ncbi.nlm.nih.gov',
            user='anonymous',
            passwd=secrets.getsecret('email')
        )
        ftp.cwd(ftplocation)
        filenames = ftp.nlst()
        for filename in filenames:
            localfilename = os.path.join(self.location, filename)
            # md5 logic
            if check_md5 and localfilename.endswith('.xml.gz'):
                try:
                    md5 = open(localfilename+'.md5').read().strip().split()[1]
                except FileNotFoundError:
                    refetch = False
                try:
                    with open(localfilename, 'rb') as file_to_check:
                        md5_returned = hashlib.md5(file_to_check.read()).hexdigest()
                        if md5 != md5_returned:
                            refetch = True
                        else:
                            refetch = False
                except FileNotFoundError:
                    refetch = True
            else:
                refetch = False
            if refetch or not os.path.exists(localfilename):
                print('Retrieving', filename)
                with open(localfilename, 'wb') as fh:
                    ftp.retrbinary('RETR ' + filename, fh.write)
        ftp.close()  # ftp.quit()

    def update(self):
        """Retrieve new documents and process them.
        Could be used to start from scratch.

        Updated pubmed xml documents contain new, revised and deleted abstracts,
        and should be handled accordingly.
        """
        self.retrieve_documents(check_md5=True)
        self.retrieve_documents(ftplocation='pubmed/updatefiles', check_md5=True)

    def process_documents(self, callback, *args, verbose=False, progress=True, onepass=False, limit=None, **kwargs):
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
        import hashlib
        import json
        xmlfilenames = sorted(
            glob.glob(os.path.join(self.location, '*.xml.gz'))
        )
        if limit:
            import random
            xmlfilenames = random.sample(xmlfilenames, limit)
        totalxmlfiles = len(xmlfilenames)
        article_pos = 0
        if onepass:
            try:
                processedFiles = set(json.load(open(os.path.join(self.location, onepass))))
            except FileNotFoundError:
                processedFiles = set()
        for xmli, xmlfilename in enumerate(xmlfilenames):
            if onepass and os.path.basename(xmlfilename) in processedFiles:
                continue
            # check md5
            md5 = open(xmlfilename+'.md5').read().strip().split()[1]
            with open(xmlfilename, 'rb') as file_to_check:
                md5_returned = hashlib.md5(file_to_check.read()).hexdigest()
                if md5 != md5_returned:
                    print('Run the method `pmc.retrieve_documents(check_md5=True)` on the pubmed object')
                    raise FileNotFoundError('Correct file was not downloaded', xmlfilename)
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
                            callback(title, abstract, elem, (xmli, article_pos, pmid), *args, **kwargs)  # TODO save map xmli to filename
                        except AttributeError as e:
                            if verbose:
                                print(e)
                        article_pos += 1
                        # TODO output article
                        root.clear()  # ensures only one article is kept in memory
                        # but because of the 'clear' you cannot use the root for info on
                        # current iterated PubmedArticle element
            if progress:
                print(end='\rProgress (%): {:.4f}'.format((xmli+1)/totalxmlfiles)*100)
            if onepass:
                processedFiles.add(os.path.basename(xmlfilename))
        if onepass:
            json.dump(list(processedFiles), open(os.path.join(self.location, onepass), 'wt'))

    def build_document_index(self, shards=10, include_mesh=True, limit=None):
        """Build an index for fast document retrieval with multiprocessing (one process/shard)

        shards (int): The number of document partitionings to use. Whoosh has memory issues
          for large indexes, this is a way around.
        include_mesh (bool): Include mesh terms in shelve db.
        limit (int): If limit, stop indexing after limit number for testing.
        """
        import multiprocessing as mp
        queues = [mp.Queue(maxsize=1000) for i in range(shards)]
        indexdir = os.path.join(self.location, '.index')
        if not os.path.exists(indexdir):
            os.mkdir(indexdir)

        def worker_function(shardnumber):
            from whoosh.index import create_in, open_dir
            from whoosh import fields
            import datetime
            import shelve
            import warnings

            schema = fields.Schema(
                title=fields.TEXT(stored=True),
                pmid=fields.ID(stored=True, unique=True),
                content=fields.TEXT(stored=True),
                date=fields.DATETIME(stored=True),
                # filepos=fields.INT(),
                # articlepos=fields.INT()
            )

            if not os.path.exists(os.path.join(indexdir, str(shardnumber))):
                os.mkdir(os.path.join(indexdir, str(shardnumber)))
                ix = create_in(os.path.join(indexdir, str(shardnumber)), schema)
            else:
                ix = open_dir(os.path.join(indexdir, str(shardnumber)), schema=schema)

            # shelve <> for time issues avoiding create sqlite db
            dbshelve = shelve.open(os.path.join(indexdir, str(shardnumber), 'pmid.shelve'))
            # conn = sqlite3.connect(
            #    os.path.join(indexdir,str(shardnumber),'pmid.db'),
            #    detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES
            # )
            # dbcursor = conn.cursor()
            # if not(dbcursor.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchone()):
            #    # Create table if this is first connection
            #    dbcursor.execute('''CREATE TABLE abstracts
            #        (pmid INTEGER PRIMARY KEY, version INTEGER, date TIMESTAMP, filepos INTEGER, articlepos INTEGER, xmlbytes INTEGER)'''
            #    )
            #    conn.commit()

            import calendar
            import locale
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            # for available settings see $ locale -a
            # if en_US.UTF-8 this will fail
            months = dict(
                (v, k) for k, v in list(enumerate(calendar.month_abbr))+list(enumerate(calendar.month_name))
            )

            commitCounter = 0
            commitLoop = 1000  # do a commit every 1000 document inserts
            while True:
                title, abstract, elem, position = queues[shardnumber].get()
                if not elem:
                    break

                if elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Year') is not None:
                    # First get month as sometimes it is a number, sometimes 3 letters
                    datemonth = elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Month')
                    if datemonth is not None:
                        datemonth = int(datemonth.text) if datemonth.text.isnumeric() else months[datemonth.text]
                    else:
                        datemonth = 1
                    try:
                        date = datetime.datetime(
                            int(elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Year').text),
                            datemonth,
                            int(elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Day').text) if
                            elem.find('MedlineCitation/Article/Journal/JournalIssue/PubDate/Day') is not None else 1
                            # some PubDate s miss the 'Month' and/or 'Day'
                        )
                    except ValueError as e:
                        # Only catching ValueError here, MedlineCitation dates should be curated by pubmed
                        print(position, e)
                        date = datetime.datetime(1, 1, 1)
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
                    date = datetime.datetime(1, 1, 1)  # Dummy date if no date available
                pmidversion = int(elem.find('MedlineCitation/PMID').get('Version'))

                # Extract mesh terms
                if include_mesh:
                    meshterms = [
                        mh.find('DescriptorName').get('UI')  # .text for the actual mesh concept
                        for mh in elem.find('MedlineCitation/MeshHeadingList')
                    ] if elem.find('MedlineCitation/MeshHeadingList') is not None else []

                # Prepare indexer writer
                if not (commitCounter % commitLoop):
                    writer = ix.writer()
                commitCounter += 1  # this ensures that creating writer and committing occur in subsequent loops

                # Check if article has already been indexed or needs updating
                # try:
                #    dbcursor.execute(
                #        'INSERT INTO abstracts(pmid,date,filepos,articlepos) values (?,?,?,?)',
                #        (position[2],date,position[0],position[1])
                #    )
                pmidstr = str(position[2])
                if pmidstr not in dbshelve:
                    dbshelve[pmidstr] = (position[0], position[1], None, date) + ((meshterms,) if include_mesh else ())
                # except sqlite3.IntegrityError:
                #    prevpos = dbcursor.execute(
                #        'SELECT filepos,articlepos,version FROM abstracts WHERE pmid=?',(position[2],)
                #    ).fetchone()
                else:
                    prevpos = dbshelve[pmidstr]
                    if prevpos[0] == position[0]:
                        if prevpos[1] == position[1]:
                            continue  # abstract already indexed
                        elif prevpos[2] == pmidversion:
                            warnings.warn('same versioned pmid ({}) in same file ({}) twice'.format(position[2], position[0]))
                    elif prevpos[0] > position[0]:
                        continue  # has already been updated in a previous run
                    # should be case where update is required
                    # dbcursor.execute(
                    #    'UPDATE abstracts SET date=?, filepos=?, articlepos=?, version=? WHERE pmid=?',
                    #    # in current setup having a pmidversion of 1 in db indicates at least one update
                    #    (date,position[0],position[1],pmidversion,position[2])
                    # )
                    dbshelve[pmidstr] = (position[0], position[1], pmidversion, date) + ((meshterms,) if include_mesh else ())
                    writer.delete_by_term('pmid', str(position[2]))

                # Indexer code
                writer.add_document(
                    title=title,
                    pmid=elem.find('MedlineCitation/PMID').text,
                    # Including the title in the content so it is also indexed searched
                    content=abstract,  # title+(' ' if title.endswith('.') else '. ')+abstract,
                    date=date
                )
                if not (commitCounter % commitLoop):
                    writer.commit()
                    # conn.commit()
            if (commitCounter % commitLoop):
                # last commit if not committed in last loop
                writer.commit()
                # conn.commit()
            ix.close()
            # conn.close()
            dbshelve.close()

        def commit_abstracts(title, abstract, elem, position):
            queues[position[2] % shards].put([title, abstract, elem, position])
            # position[2] == pmid

        # Start up workers
        processes = [mp.Process(target=worker_function, args=(i,)) for i in range(shards)]
        for p in processes:
            p.start()
        # Build index
        self.process_documents(commit_abstracts, onepass='.indexed_files.json', limit=limit)
        # Wait for all workers to finish
        for q in queues:
            q.put([None, None, None, None])  # putting stop signal in each queue
        for p in processes:
            p.join()

    def query_document_index(self, query, sortbydate=False):
        """Query the corpus index

        Args:
            query (str): the elasticv-search like query str to run.
            sortbydate (bool): If set to True, might encounter memory
              issues with the current version of whoosh.
        """
        import whoosh.index as index
        from whoosh.qparser import QueryParser
        indexdirs = os.path.join(self.location, '.index', '*')
        total_idirs = len(glob.glob(indexdirs))
        results = []
        querystr = query
        print()
        for indexdir in range(total_idirs):
            indexdir = os.path.join(self.location, '.index', str(indexdir))
            print('\rProcessing shard %s/%s' % (os.path.basename(indexdir), total_idirs-1), end='')
            ix = index.open_dir(indexdir)
            query = QueryParser("content", ix.schema).parse(querystr)
            with ix.searcher() as searcher:
                search_results = searcher.search(query, limit=None, scored=False, sortedby='date' if sortbydate else None)
                results += [r.fields() for r in search_results]
        print('\n', len(results), 'retrieved')
        return results

    def filter_documents(self, regex):
        """returns the filtered set of documents matching regex"""
        # TODO should be based on lucene index search
        import re
        # if regex is provided as str compile as regex
        regex = regex if isinstance(regex, re.Pattern) else re.compile(regex, re.IGNORECASE)

        def callback(title, abstract, elem, position, filteredList):
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
        import re
        import shelve  # , sqlite3
        xmlfilenames = sorted(
            glob.glob(os.path.join(self.location, '*.xml.gz'))
        )
        articlere = re.compile(b'<PubmedArticle>.+?</PubmedArticle>', re.MULTILINE | re.DOTALL)
        pmidre = re.compile(b'<PMID.+?>(.+)</PMID>')
        locshelve = shelve.open(os.path.join(self.location, 'pmid_locations.shelve'))
        # Chose a shelve instead of sqlite db for time reasons
        #  In: %timeit locshelve[pmid]
        #  2.96 µs ± 462 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        #  In: %timeit c.execute('SELECT * FROM ablocations WHERE pmid=?', (pmid,)).fetchone()
        #  5.72 s ± 38.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # conn = sqlite3.connect(os.path.join(self.location,'pmid_locations.db'))
        # c = conn.cursor()
        # c.execute('''CREATE TABLE ablocations
        #     (pmid text, filename text, start integer, length integer)''')
        # conn.commit()
        for i, xmlfilename in enumerate(xmlfilenames):
            print('\r', i, end='', sep='')
            basename = os.path.basename(xmlfilename)
            with gzip.open(xmlfilename) as xmlfilehandler:
                xmlcontent = xmlfilehandler.read()
                for article in articlere.finditer(xmlcontent):
                    pmid = pmidre.search(article.group()).groups()[0].decode()
                    locshelve[pmid] = (basename, article.start(), article.end()-article.start())
                    # c.execute(
                    #    "INSERT INTO ablocations VALUES ('?','?',?,?)",
                    #    (pmid, basename, article.start(), article.end()-article.start())
                    # )
            # conn.commit()
        # conn.close()
        locshelve.close()

    def retrieve_article_xmls(self, pmids):
        """Retrieve the xml of a set of articles

        Args:
            pmids (list): List or any iterable of pmid ids in str format

        Returns:
            dict of xml trees: pmid key and xml tree value for each article requested
        """
        import shelve  # sqlite3
        locshelve = shelve.open(os.path.join(self.location, 'pmid_locations.shelve'))
        # conn = sqlite3.connect(os.path.join(self.location,'pmid_locations.db'))
        articles = {}
        # c = conn.cursor()
        for pmid in pmids:
            # c.execute('SELECT * FROM ablocations WHERE pmid=?', (pmid,))
            abref = (pmid,) + locshelve[pmid]  # c.fetchone()
            with gzip.open(os.path.join(self.location, abref[1])) as xmlfilehandler:
                xmlfilehandler.seek(abref[2])
                xmlcontent = xmlfilehandler.read(abref[3])
                articles[pmid] = ET.fromstring(xmlcontent)
        locshelve.close()
        # conn.close()
        return articles


class PubmedCentralCollection(BaseDocumentCollection):
    """Example:

    >>> pmc = PubmedCentralCollection('pubmedcentral')
    >>> pmc.retrieve_documents()
    """

    def retrieve_documents(self):
        import ftplib
        from sina.config import secrets
        ftplocations = [
            '/pub/pmc/oa_bulk',  # open access commercial and non-commercial use
            # PMID mapping file seems to be PMC-ids.csv.gz in /pub/pmc
            '/pub/pmc/manuscript',  # author manuscript collection
            '/pub/pmc/historical_ocr',  # historical ocr collection
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
            if not os.path.exists(os.path.join(self.location, collection)):
                os.mkdir(os.path.join(self.location, collection))
            for filename in filenames:
                if filename.endswith('.tmp'):
                    continue  # skip if retrieving while ncbi is making update
                localfilename = os.path.join(self.location, collection, filename)
                # no md5 s at locations for pmc
                if not os.path.exists(localfilename):
                    print('Retrieving', filename)
                    with open(localfilename, 'wb') as fh:
                        ftp.retrbinary('RETR ' + filename, fh.write)
        ftp.close()  # ftp.quit()

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
      test_fraction (float, pd.DataFrame): If float provided a random subset equaling test_fraction
        is set apart for testing purposes after machine learning tasks downstream. Can also be provided
        as `DataFrame` directly, which is useful if you need to control test cases for comparitive purposes.
      saveloc (str): If provided, dir where figures and other output are saved.
    """

    def __init__(self, corpus, results, test_fraction=.25, saveloc=False):
        import pandas as pd
        import warnings
        self.results = pd.DataFrame(results).set_index('pmid')
        duplicates = self.results.index.duplicated().sum()
        if duplicates:
            warnings.warn('Removing %s pmid duplicates' % duplicates)
            self.results = self.results.reset_index().drop_duplicates('pmid').set_index('pmid')
        nancontent = self.results[['title', 'content']].isna().sum().sum()
        if nancontent:
            warnings.warn('Removing %s records that have title/abstract missing' % nancontent)
            self.results.dropna(subset=['title', 'content'], inplace=True)
        self.corpus = corpus
        if isinstance(test_fraction, pd.DataFrame):
            self.results_test = test_fraction
            self.testset = True
        elif test_fraction:
            from sklearn.model_selection import train_test_split
            self.results, self.results_test = train_test_split(self.results, test_size=test_fraction)
            self.testset = True
        else:
            self.testset = False
        self.saveloc = saveloc
        if not os.path.exists(self.saveloc):
            os.mkdir(self.saveloc)

    def preprocess_text(self, title2content=True, bigrams=2, split_hyphens=False):
        """Preprocessing text, steps involving tokenizations and decisions on bigrams.
        The methods creates an function attribute `preprocess_documents` that can be used
        to preprocess other documents in the same way, for example an extra test set of
        documents.

        Args:
          title2content (bool): Add the title to the abstract content.
          bigrams (bool|int): if True or 1 build bigram, if 2 build trigram, etc
          split_hyphens (bool): split hyphened words into subtokens
        """
        import spacy
        import gensim
        import re
        import functools

        if title2content:
            # Currently whoosh index does not add the title to the content
            # if this changes in the future this section should be removed
            self.results.content = self.results.title + ' ' + self.results.content
            if self.testset:
                self.results_test.content = self.results_test.title + ' ' + self.results_test.content

        # Tokenization
        is_digit = re.compile(r'\d+(\.\d+)?')  # spacy.token.is_digit not picking up floats
        nlp = spacy.load('en', disable=['ner', 'parser'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        if not split_hyphens:
            # Prevent splitting intra-word hyphens
            suffixes = nlp.Defaults.suffixes + (r'''\w+-\w+''',)
            suffix_regex = spacy.util.compile_suffix_regex(suffixes)
            nlp.tokenizer.suffix_search = suffix_regex.search

        def make_documents(docs):
            return [
                [
                    [
                        token.lemma_.lower() for token in sent
                        if not (token.is_stop
                                or token.is_punct
                                or token.is_space
                                or is_digit.fullmatch(token.text))
                    ]
                    for sent in nlp(txt).sents
                ]
                for txt in docs
            ]
        documents = make_documents(self.results.content)
        phrase_models = []  # to store the bigram models
        while bigrams:
            bigrams -= 1  # for trigram set bigrams=2
            bigram = gensim.models.Phrases(
                (sent for doc in documents for sent in doc), min_count=30, progress_per=10000,
                delimiter=chr(183).encode()  # interpunct aka &middot; as delimiter
            )
            documents = [[bigram[sent] for sent in doc] for doc in documents]
            phrase_models.append(lambda s: bigram[s])

        def apply_bigrams(sent):
            return functools.reduce(
                lambda value, function: function(value),
                phrase_models,
                sent
            )
        self.preprocess_documents = lambda documents: [
            [apply_bigrams(sent) for sent in doc] for doc in make_documents(documents)
        ]
        self.results['processed'] = documents
        if self.testset:
            self.results_test['processed'] = self.preprocess_documents(self.results_test.content)

    def transform_text(self, method='counts', tfid_kwargs={}, preprocess=False, **kwargs):
        """Transform text in numerical format that can be used by predictive algorithms.
        Sets `self.X` and `self.X_test`

        Args:
          method (str): 'counts', 'tfid'
          tfid_kwargs (dict): Passed to sklearn.feature_extraction.text.TfidfTransformer
          preprocess (bool): Perform first the text preprocessing, if this is not set to true
            but kwargs are provided, preprocessing is also performed.
          **kwargs: Passed to method `preprocess_text`
        """
        import gensim
        import itertools as it
        from tensorflow.keras import preprocessing

        # Text preprocessing
        if kwargs or preprocess:
            self.preprocess_text(**kwargs)

        # Gensim corpus
        self.dictionary = gensim.corpora.Dictionary((it.chain.from_iterable(doc) for doc in self.results.processed))
        self.results['bow'] = [
            self.dictionary.doc2bow(it.chain.from_iterable(doc)) for doc in self.results.processed
        ]
        self.X = gensim.matutils.corpus2csc(self.results.bow, num_terms=len(self.dictionary)).T

        if method == 'tfid':
            from sklearn.feature_extraction.text import TfidfTransformer
            self.tfidftransformer = TfidfTransformer(**tfid_kwargs)
            self.X = self.tfidftransformer.fit_transform(self.X)
        elif method == 'idx':
            self.results['idx'] = [
                self.dictionary.doc2idx(it.chain.from_iterable(doc)) for doc in self.results.processed
            ]
            self.X = preprocessing.sequence.pad_sequences(self.results.idx, padding='post', maxlen=500, value=-1)+1

        if self.testset:
            self.results_test['bow'] = [
                self.dictionary.doc2bow(it.chain.from_iterable(doc)) for doc in self.results_test.processed
            ]
            self.X_test = gensim.matutils.corpus2csc(self.results_test.bow, num_terms=len(self.dictionary)).T
            if method == 'tfid':
                self.X_test = self.tfidftransformer.transform(self.X_test)
            elif method == 'idx':
                self.results_test['idx'] = [
                    self.dictionary.doc2idx(
                        it.chain.from_iterable(doc),
                       unknown_word_index=len(self.dictionary)
                    ) for doc in self.results_test.processed
                ]
                self.X_test = (
                    preprocessing.sequence.pad_sequences(self.results_test.idx, padding='post', maxlen=500, value=-1)+1
                ).clip(max=len(self.dictionary)+2)  # all unknown words get same value

    def analyze_mesh(self, topfreqs=20, getmeshnames=True):
        """Retrieves mesh terms for all documents
        and count the most frequent ones.

        Args:
          topfreqs (int): The top frequent mesh terms to include in the mesh table.
          getmeshnames (bool): Include human readable name of mesh terms.
        """
        from collections import Counter
        import shelve
        import pandas as pd
        shards = sorted(glob.glob(os.path.join(self.corpus.location, '.index/*')))
        lenshards = len(shards)
        pmiddbs = [
            shelve.open(
                os.path.join(self.corpus.location, '.index', str(i), 'pmid.shelve'),
                flag='r') for i in range(lenshards)
        ]
        self.meshterms = [pmiddbs[int(pmid) % lenshards][pmid][4] for pmid in self.results.index]
        if self.testset:
            self.meshterms_test = [pmiddbs[int(pmid) % lenshards][pmid][4] for pmid in self.results_test.index]
        self.meshfreqs = Counter((mt for a in self.meshterms for mt in a))

        # Filter top
        self.meshtop = pd.DataFrame(
            self.meshfreqs.most_common(topfreqs),
            columns=['meshid', 'freq']
        )
        self.topfreqs = topfreqs
        if getmeshnames:
            # wget ftp://nlmpubs.nlm.nih.gov/online/mesh/MESH_FILES/xmlmesh/desc2019.gz
            import gzip
            import sina.config
            import xml.etree.ElementTree as ET
            meshfile = os.path.join(sina.config.appdirs.user_data_dir, 'meshterms2019.gz')
            if not os.path.exists(meshfile):
                from ftplib import FTP
                ftp = FTP('nlmpubs.nlm.nih.gov')
                ftp.login(user='anonymous')
                ftp.cwd('/online/mesh/MESH_FILES/xmlmesh/')
                for rfile in ftp.nlst():
                    # searching for file with filename descXXXX.gz
                    if rfile.startswith('desc') and rfile.endswith('.gz') and len(rfile) == 11:
                        rmeshfile = rfile
                        break
                # if rmeshfile not found will raise exception here
                logging.info('downloading mesh terms table file %s form nih.gov...', rmeshfile)
                with open(meshfile, 'wb') as localfile:
                    ftp.retrbinary('RETR '+rmeshfile, localfile.write, 1024)
                ftp.quit()
                logging.info('download completed')
            with gzip.open(meshfile) as xmlfilehandle:
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
            {pmid: meshtop_ix.isin(self.meshterms[i]) for i, pmid in enumerate(self.results.index)}
        ).T
        self.meshtabel['mtindexsum'] = self.meshtabel.sum(axis=1)
        self.meshtabel.sort_values('mtindexsum', ascending=False, inplace=True)
        del self.meshtabel['mtindexsum']
        if self.testset:
            self.meshtabel_test = pd.DataFrame(
                {pmid: meshtop_ix.isin(self.meshterms_test[i]) for i, pmid in enumerate(self.results_test.index)}
            ).T

            # Direct links to the arrays for prediction algorithms
            self.Y = self.meshtabel.values
            self.Y_test = self.meshtabel_test.values

    def predict_meshterms(
            self, method='kmeans_summ', kfilter=100, model='bayes', idcf=False,
            kmeans_only_freqs=False, embedding=None, rebalance=False
    ):
        """Using classical ML algorithms for predicting
        mesh terms belonging to an article

        Requires:
          X, X_test, Y, and Y_test to have been set as attributes on the model.
          This can be done with the methods `transform_text` and `analyze_mesh`.

        Args:
          method (str): options
            'tfidf_wv': Experiment to process in a similar
               way to -> http://xplordat.com/2018/12/14/want-to-cluster-text-try-custom-word-embeddings/
            'kmeans_summ': Summarize abstracts using kmeans in wordembedding space
          kfilter (int): If provided the number to reduce X to for k best predicting features.
          model (str): ML model to use, options: 'bayes', 'svm', 'sgd', 'logreg'
          idcf (bool): Use inverse document cluster frequencies.
          kmeans_only_freqs (bool): Only process k-cluster frequencies
          embedding (PubmedQueryResult): Use embedding from external
            PubmedQueryResult if provided.
          rebalance (bool): Balance dataset by random undersampling.
        """
        import numpy as np
        from sklearn import svm, naive_bayes, metrics
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import SGDClassifier

        # Preprocessing only
        if method == 'tfidf_wv':
            # TODO move to transform_text
            preptext = Pipeline([
                ('vect', CountVectorizer(token_pattern=r'\S+')),  # only splitting as already preprocessed
                ('tfidf', TfidfTransformer())
            ])
            preptext.fit(self.results.processed)
            X = preptext.transform(self.results.processed)
            X_test = preptext.transform(self.results_test.processed)

            # Transform to tfidf_wv
            embedding = self.embedding if embedding else self.embedding
            vocab = sorted(preptext.steps[0][1].vocabulary_)
            embeddedword = [w in embedding.wv.vocab for w in vocab]
            wvs = np.hstack(
                [embedding.wv[w] for w in vocab if w in embedding.wv.vocab]
            )
            X = np.vstack(
                [
                    np.multiply(wvs, np.repeat(x.todense()[:, embeddedword], embedding.wv.vector_size))
                    for x in X
                ]
            )
            X_test = np.vstack(
                [
                    np.multiply(wvs, np.repeat(x.todense()[:, embeddedword], embedding.wv.vector_size))
                    for x in X_test
                ]
            )

        elif method == 'kmeans_summ':
            # Naive Bayes
            # model = Pipeline([('vect', CountVectorizer()),
            #        ('tfidf', TfidfTransformer()),
            #        ('clf', naive_bayes.MultinomialNB()),
            # ])
            # Logistic regression
            # model = Pipeline([('vect', CountVectorizer()),
            #            ('tfidf', TfidfTransformer()),
            #            ('clf', LogisticRegression(n_jobs=1, C=1e5)),
            # ])
            # Linear SVM
            # sgd = Pipeline([('vect', CountVectorizer()),
            #        ('tfidf', TfidfTransformer()),
            #        ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
            #       ])

            # Prepare data
            X = self.X
            X_test = self.X_test
            # X = np.vstack(
            #    [self.normalize_text_length(list(it.chain.from_iterable(doc)), idcf, kmeans_only_freqs, embedding) for doc in self.results.processed]
            # )
            # X_test = np.vstack(
            #    [self.normalize_text_length(list(it.chain.from_iterable(doc)), idcf, kmeans_only_freqs, embedding) for doc in self.results_test.processed]
            # )
        else:
            raise Exception('No method', method)

        # ML model
        self.models = []
        train_accs = []  # train accuracies
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        if kfilter:
            k = kfilter
        for i in range(self.Y.shape[1]):
            y = self.Y[:, i]

            # Balance
            # For other resampling options:
            # https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
            if rebalance == 'undersample':
                from imblearn.under_sampling import RandomUnderSampler
                rus = RandomUnderSampler(return_indices=True)
                X_rsmpl, y, id_rsmpl = rus.fit_sample(X, y)
            elif rebalance == 'oversample':
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(ratio='minority')
                X_rsmpl, y = smote.fit_sample(X, y)
            else:
                X_rsmpl = X

            # Filter
            if kfilter:
                from sklearn.feature_selection import SelectKBest, f_classif
                # other options SelectPercentile or GenericUnivariateSelect for grid search
                # For regression: f_regression, mutual_info_regression
                # For classification: chi2, f_classif, mutual_info_classif
                kfilter = SelectKBest(f_classif, k=k)
                X_fil = kfilter.fit_transform(X_rsmpl, y)
                X_fil_test = kfilter.transform(X_test)
            else:
                X_fil, X_fil_test = X, X_test

            # Model
            if model == 'bayes':
                model = naive_bayes.MultinomialNB()  # BernoulliNB() #GaussianNB()
            elif model == 'svm':
                model = svm.SVC(kernel='rbf', class_weight='balanced', gamma='scale')
            elif model == 'logreg':
                model = LogisticRegression(n_jobs=1, C=1e5)
            elif model == 'sgd':
                model = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
            model.fit(X_fil, y)
            Y_test_pred = model.predict(X_fil_test)
            Y_train_pred = model.predict(X_fil)
            accuracies.append(metrics.accuracy_score(self.Y_test[:, i], Y_test_pred))
            train_accs.append(metrics.accuracy_score(y, Y_train_pred))
            precisions.append(metrics.precision_score(self.Y_test[:, i], Y_test_pred))
            recalls.append(metrics.recall_score(self.Y_test[:, i], Y_test_pred))
            f1s.append(metrics.f1_score(self.Y_test[:, i], Y_test_pred))
            print(i, train_accs[-1], accuracies[-1])
            self.models.append(model)
        self.meshtop['train_acc'] = train_accs
        self.meshtop['pred_acc'] = accuracies
        self.meshtop['pred_prec'] = precisions
        self.meshtop['pred_rec'] = recalls
        self.meshtop['pred_F1'] = f1s
        # for testing purposes
        # X_fil will only be relevant for last tested model
        self.X_fil, self.X_fil_test = X_fil, X_fil_test
        self.kfilter = kfilter

    def nn_keras_predictor(
            self, model='simple', learning_rate=0.01, epochs=5, batch_size=10,
            num_filters=128, kernel_size=5, embedding=None,
            embedding_trainable=False, plothist=True, return_model=False
    ):
        """Implementing multi-label classification
        not multi-class classification

        Args:
          model (str): 'simple', 'cnn'
          learning_rate (float): nn model learning rate.
          epochs (int): Number of epochs.
          batch_size (int): Number per batch.
          num_filters (int): CNN filters.
          kernel_size (int): CNN kernel size.
          embedding (gensim.embedding): Use externally trained embedding.
          embedding_trainable (bool): Embedding can be further optimized if true.
          plothist (bool): plot the loss and accuracy training evolution.
          return_model (bool): If true, just return model; used for grid search.
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Activation, Dropout
        # from tensorflow.keras.preprocessing import text, sequence
        # from tensorflow.keras import Input, layers, optimizers, utils
        from tensorflow.keras import layers
        import numpy as np
        import pandas as pd
        from sklearn import metrics
        # learning_rate = 0.01
        # lr_decay = learning_rate/epochs  # previously tested values: 1e-6

        # Preprocess data
        # if textprep:
        #     max_words = 100000
        #     max_len = 300 #usual max abstract length
        #     tokenizer = text.Tokenizer(num_words=max_words, lower=True)
        #     tokenizer.fit_on_texts(self.results.content)
        #     self.X = tokenizer.texts_to_sequences(self.results.content)
        #     self.X = sequence.pad_sequences(self.X, maxlen=max_len)
        #     self.X_test = tokenizer.texts_to_sequences(self.results_test.content)
        #     self.X_test = sequence.pad_sequences(self.X_test, maxlen=max_len)

        # Prepare embedding matrix
        embedding = embedding if embedding else self.embedding
        embedding_vector_size = self.embedding.wv.vector_size
        embedding_mx = np.zeros((
            len(self.dictionary)+2,  # two larger than dict for padding and unknown words
            embedding_vector_size
        ))
        for w in embedding.wv.vocab:  # iterate words that have a calculated word vector
            try:
                embedding_mx[self.dictionary.token2id[w]+1] = embedding.wv[w]  # +1 as 0-index reserved for padding
            except KeyError:
                if w not in self.dictionary.token2id:
                    continue

        # Build the model
        if model == 'simple':
            model = Sequential()
            model.add(Dense(512, input_shape=(self.X.shape[1],)))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.Y.shape[1]))  # number of classes
            model.add(Activation('sigmoid'))  # sigmoid instead of softmax for multi-label prediction
        elif model == 'cnn':
            model = Sequential()
            model.add(layers.Embedding(
                input_dim=len(self.dictionary)+2,  # two larger than dict for padding and unknown words
                output_dim=embedding_vector_size,  # embedding dimension -> vector size
                weights=[embedding_mx],
                input_length=self.X.shape[1],
                trainable=embedding_trainable
            ))
            model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
            model.add(layers.GlobalMaxPool1D())  # model.add(layers.Flatten())
            model.add(layers.Dense(10, activation='relu'))
            model.add(layers.Dense(self.Y.shape[1], activation='sigmoid'))

        # sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        # adam = optimizers.Adam(lr=learning_rate, decay=lr_decay)
        model.compile(loss='binary_crossentropy',  # 'binary_crossentropy' 'mean_squared_error' 'categorical_crossentropy', # categorical didn't give good results
                      optimizer='adam',  # sgd, adam,
                      metrics=['accuracy'])

        if return_model:  # for grid search
            return model
        else:
            model.summary()

        mh = model.fit(
            self.X, self.Y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=0.1,
            class_weight=self.Y.sum(axis=0)/self.Y.sum(axis=0).min()  # 'auto', # weight for important classes
            # interesting solution @ https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras
            # sample_weight=... # weight for important samples
        )
        Y_pred_prob_test = model.predict(self.X_test)
        # Calculate accuracy/mesh_term
        Y_pred_test = Y_pred_prob_test > .5  # consider assigned label if above .5 'chance'
        Y_accuracies_test = (
            (self.Y_test & Y_pred_test) | (~self.Y_test & ~Y_pred_test)
        ).mean(axis=0)
        score = model.evaluate(
            self.X_test, self.Y_test,
            batch_size=batch_size, verbose=1)
        print('Test accuracy:', score[1], Y_accuracies_test)
        # Performance tabel
        self.meshtop_nn = pd.DataFrame(
            np.vstack(
                metrics.classification.precision_recall_fscore_support(self.Y_test, Y_pred_test)
            ).T,
            columns=['prec', 'rec', 'F1', 'support'],
            index=self.meshtop.meshid
        )

        if plothist:
            import matplotlib.pyplot as plt
            import numpy as np
            fig, ax = plt.subplots()
            ax.plot(np.arange(0, epochs), mh.history["loss"], label="train_loss")
            ax.plot(np.arange(0, epochs), mh.history["val_loss"], label="val_loss")
            ax.plot(np.arange(0, epochs), mh.history["accuracy"], label="train_acc")
            ax.plot(np.arange(0, epochs), mh.history["val_accuracy"], label="val_acc")
            ax.set_title("Training Loss and Accuracy")
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("Loss/Accuracy")
            ax.legend(loc="upper left")
            if self.saveloc:
                fig.savefig(os.path.join(self.saveloc, 'train_history.pdf'))
        return model

    def nn_grid_search(self, external_embedding, epochs=5,
                       randomized_search=False, n_jobs=5):
        """Comparing nn architectures and exploring
        hyperparamter combinations

        Args:
          external_embedding (gensim.embedding): External embedding to use in comparison.
          epochs (int): Neural network epochs.
          randomized_search (int): False or int indicating amount of
            parameter combinations to search (e.g. 10)
          n_jobs (int): Number of parallel jobs for executing grid.
        """
        from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        import numpy as np
        import pandas as pd

        print('Grid search for embeddings/hyperparameters')
        # Parameter grid for grid search
        param_grid = dict(
            num_filters=[64, 128],  # only selected two out of [32, 64, 128] to limit combis
            kernel_size=[5, 7],  # only selected two out of [3, 5, 7] to limit combis
            embedding=[self.embedding] + external_embedding
            if isinstance(external_embedding, list) else [external_embedding],
            embedding_trainable=[False, True],
            model=['cnn'],
            return_model=[True]
        )

        model = KerasClassifier(
            build_fn=self.nn_keras_predictor,
            epochs=epochs, batch_size=10,
            verbose=False
        )

        grid = RandomizedSearchCV(
            estimator=model, param_distributions=param_grid,
            cv=4, verbose=1, n_iter=randomized_search, n_jobs=n_jobs
        ) if randomized_search else GridSearchCV(
            estimator=model, param_grid=param_grid,
            cv=4, verbose=1, n_jobs=n_jobs
        )

        # Not attaching raw nn_grid_result to self
        # as it generates unpickable _thread._local objects
        nn_grid_result = grid.fit(self.X, self.Y)
        self.nn_best_params = nn_grid_result.best_params_
        self.nn_best_score = nn_grid_result.best_score_
        test_accuracy = grid.score(self.X_test, self.Y_test)
        print(
            'Best Accuracy : {:.4f}\n{}\nTest Accuracy : {:.4f}\n\n'.format(
                self.nn_best_score,
                self.nn_best_params,
                test_accuracy
            )
        )

        # Reformat grid results
        self.nn_grid_result = pd.DataFrame(nn_grid_result.cv_results_)
        del self.nn_grid_result['params']
        self.nn_grid_result['vocab_size'] = self.nn_grid_result.param_embedding.apply(
            lambda x: len(
                x.vocab if hasattr(x, 'vocab') else x.wv.vocab
            )
        )
        self.nn_grid_result['param_embedding'] = self.nn_grid_result['param_embedding'].apply(
            lambda x: param_grid['embedding'].index(x)
        )

        # Calculate std deviation on grid results that have same
        # settings except for embedding
        self.nn_grid_embedding_std = np.std(
            nn_grid_result.cv_results_['mean_test_score'][
                # Selector best settings except for embedding
                (nn_grid_result.cv_results_['param_embedding_trainable'] ==
                 nn_grid_result.best_params_['embedding_trainable']) &
                (nn_grid_result.cv_results_['param_kernel_size'] ==
                 nn_grid_result.best_params_['kernel_size']) &
                (nn_grid_result.cv_results_['param_num_filters'] ==
                 nn_grid_result.best_params_['num_filters'])
            ]
        )

    def gensim_w2v(self, vecsize=100):
        """Build word2vec model with gensim

        Args:
          vecsize (int): Size of the embedding vectors
        """
        import gensim
        w2model = gensim.models.word2vec.Word2Vec([sent for doc in self.results.processed for sent in doc], size=vecsize, hs=1)

        # Only retaining embedded words in results processed section TODO MAKE OPTIONAL
        # self.results.processed = [' '.join([t for t in doc if t in w2model.wv.vocab]) for doc in self.results.processed]
        # if self.testset:
        #    self.results_test.processed = [' '.join([t for t in doc if t in w2model.wv.vocab]) for doc in self.results_test.processed]

        self.embedding = w2model

        # Get 100 most frequent words
        top100words = w2model.wv.index2entity[:100]
        fig = self.vizualize_embedding(top100words)

        if self.saveloc:
            fig.savefig(os.path.join(self.saveloc, 'embedding.pdf'))

        return w2model

    def vizualize_embedding(self, words, embedding=None):
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        if not embedding:
            embedding = self.embedding
        vecsize = embedding.wv.vector_size

        # Retreive word vectors
        word_vectors = embedding.wv[words]

        # Reduce with PCA dimensionality as tSNE does not scale well
        word_vectors_reduc = PCA(
            n_components=len(words) if len(words) < 50 else 50
        ).fit_transform(word_vectors) if vecsize > 50 else word_vectors
        Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(word_vectors_reduc)

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(Y[:, 0], Y[:, 1])
        for word, vec in zip(words, Y):
            ax.text(vec[0], vec[1],
                    word.title(),
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    size='medium',
                    ).set_size(15)

        return fig

    def k_means_embedding(self, k=100, calc_idcf=True, verbose=False):
        """Calculate k means centroids
        for the word embedding space

        Args:
          k (int): Number of centroids, defaults to 100.
          calc_idcf (bool): Calculate inverse document cluster frequencies.
          verbose (bool): Print closest word vectors to calculated centroids.
        """
        # from sklearn.neighbors import KNeighborsClassifier
        from sklearn.cluster import KMeans
        # setting n_jobs to -1 to use all available processors
        kmeans = KMeans(k, init='k-means++', n_jobs=-1)
        kmeans.fit(self.embedding.wv[self.embedding.wv.index2entity])
        if calc_idcf:
            import pandas as pd
            predictions = pd.DataFrame({
                'label': kmeans.predict(self.embedding.wv[self.embedding.wv.index2entity]),
                'word': self.embedding.wv.index2entity
            })
            predictions['doc_freq'] = predictions.word.apply(
                lambda x: {
                    # joining sentences in docs
                    i for i, d in enumerate((([token for sent in doc for token in sent] for doc in self.results.processed))) if x in d
                }
            )
            if verbose:
                print(predictions.value_counts())
            # Calculate inverse document cluster-frequencies
            self.idcf = predictions.groupby(
                'label'
            ).apply(lambda x: 1/len(set.union(*x.doc_freq)))

        if verbose:
            for i, center in enumerate(kmeans.cluster_centers_):
                print(self.embedding.wv.similar_by_vector(center)[0])
        self.embedding_kmeans = kmeans

    def gensim_topics(self, num_topics=10):
        import gensim
        import pandas as pd
        dictionary = gensim.corpora.Dictionary(self.results.processed)
        corpus_bow = [dictionary.doc2bow(c) for c in self.results.processed]
        ldamodel = gensim.models.LdaModel(
            corpus=corpus_bow, num_topics=num_topics, id2word=dictionary, iterations=100
        )
        self.corpus_lda_topics = pd.concat([
            pd.DataFrame(d, columns=['topic', 'prob']).set_index('topic').T
            for d in ldamodel[corpus_bow]
        ]).reset_index(drop=True).fillna(0)
        # test set
        if self.testset:
            corpus_bow_test = [dictionary.doc2bow(c) for c in self.results_test.processed]
            self.corpus_lda_topics_test = pd.concat([
                pd.DataFrame(d, columns=['topic', 'prob']).set_index('topic').T
                for d in ldamodel[corpus_bow_test]
            ]).reset_index(drop=True).fillna(0)
        return ldamodel

    def normalize_text_length(self, textlist, idcf=True, only_freqs=False, external=None):
        """Using k centroids generated by self.k_means_embedding
        transform text to average vectors with vector count per cluster

        Args:
          textlist (list): A list of processed tokens representing the text.
            Should be processed in the same manner as the embedding.
          idcf (bool): Calculate cluster frequence inverse document cluster frequency,
            kmeans clusters should have been generated with calc_idcf=True.
            In current implementation idcf does not return embedding vectors.
          only_freqs (bool): Do not include wordvectors.
          external (PubmedQueryResult): Use embedding from external
            PubmedQueryResult if provided.
        """
        import pandas as pd
        import numpy as np
        # Set embedding to use
        embedding = external.embedding if external else self.embedding
        embedding_kmeans = external.embedding_kmeans if external else self.embedding_kmeans
        textvectors = [
            embedding.wv[token] for token in textlist if token in embedding.wv
        ]
        predictions = pd.DataFrame({
            'label': embedding_kmeans.predict(textvectors),
            'wordvector': textvectors
        })
        clustercounts = predictions.groupby('label').count()
        if idcf:
            idcf = external.idcf if external else self.idcf
            clustercounts.wordvector = (clustercounts.wordvector*idcf).fillna(0)
            return clustercounts.wordvector.values
        # normalize for length of abstract (only if not idcf)
        # TODO is not ideal if then processing with e.g. naive_bayes.MultinomialNB
        elif clustercounts.wordvector.sum():
            clustercounts.wordvector = clustercounts.wordvector/clustercounts.wordvector.sum()
        clustercounts['avgvec'] = predictions.groupby('label').apply(
            lambda x: np.vstack(x.wordvector).mean(axis=0)
        )
        emptyclusters = pd.DataFrame({
            i: {'wordvector': 0, 'avgvec': np.zeros(embedding_kmeans.cluster_centers_.shape[1])}
            for i in range(embedding_kmeans.cluster_centers_.shape[0]) if i not in clustercounts.index
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

    def embedding_projection(self, emb1, emb2, topvoc_emb2=None, test_similarity=False, custom=True):
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
        import gensim
        import numpy as np
        import random
        emb2vocab = list(emb2.wv.vocab)[:topvoc_emb2] if topvoc_emb2 else emb2.wv.vocab

        # Allows projecting known words from emb1 into emb2 even if unkown
        if custom:
            commonwords = list(set(emb1.wv.vocab) & set(emb2vocab))
            sourcevec = emb1.wv[commonwords]
            targetvec = emb2.wv[commonwords]
            n, dim = targetvec.shape
            centeredS = sourcevec - sourcevec.mean(axis=0)
            centeredT = targetvec - targetvec.mean(axis=0)
            C = np.dot(np.transpose(centeredS), centeredT) / n
            V, S, W = np.linalg.svd(C)
            d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
            if d:
                S[-1] = -S[-1]
                V[:, -1] = -V[:, -1]
            R = np.dot(V, W)
            varS = np.var(sourcevec, axis=0).sum()
            c = 1/varS * np.sum(S)  # scale factor
            t = targetvec.mean(axis=0) - sourcevec.mean(axis=0).dot(c*R)
            # np.allclose(sourcevec.dot(c*R) + t, targetvec)
            err = ((sourcevec.dot(c * R) + t - targetvec) ** 2).sum()
            print('Error on training points:', err)

            def projfunc(x):
                return emb1.wv[x].dot(c * R) + t
        else:
            commonwords = [(w, w) for w in set(emb1.wv.vocab) & set(emb2vocab)]
            transmat = gensim.models.translation_matrix.TranslationMatrix(emb1.wv, emb2.wv)
            transmat.train(commonwords)
            # emb2.wv.similar_by_vector( np.dot(transmat.translation_matrix,emb1.wv['cyclin-dependent']),topn=3)
            # ==
            # transmat.translate(['cyclin-dependent'], topn=3)

            def projfunc(x):
                return np.dot(transmat.translation_matrix, emb1.wv[x])

        if test_similarity and topvoc_emb2:
            # For the moment only sensible to test if restricting reference points emb2
            from scipy import spatial
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            controlset = random.sample(list(emb2.wv.vocab)[topvoc_emb2:], topvoc_emb2)
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
            # emb training set
            Y = pca.transform(emb2.wv[emb2vocab])
            Y_projection = pca.transform(
                np.stack([projfunc(w) for w in emb2vocab if w in emb1.wv.vocab])
            )
            ax.scatter(Y[:, 0], Y[:, 1], label='original_trained')
            ax.scatter(Y_projection[:, 0], Y_projection[:, 1], label='projection_trained')
            for word, vec in list(zip(emb2vocab, Y))+list(zip(emb2vocab, Y_projection)):
                ax.text(vec[0], vec[1],
                        word.title(),
                        ).set_size(15)
            # control set
            Y_control = pca.transform(emb2.wv[controlset])
            Y_control_projection = pca.transform(
                np.stack([projfunc(w) for w in controlset if w in emb1.wv.vocab])
            )
            ax.scatter(Y_control[:, 0], Y_control[:, 1], label='original_control')
            ax.scatter(Y_control_projection[:, 0], Y_control_projection[:, 1], label='projection_control')
            for word, vec in list(zip(controlset, Y_control))+list(zip(controlset, Y_control_projection)):
                ax.text(vec[0], vec[1],
                        word.title(),
                        ).set_size(15)
            # legend
            ax.legend()
            if self.saveloc:
                fig.savefig(os.path.join(self.saveloc, 'projection.pdf'))

        return projfunc
