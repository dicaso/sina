# -*- coding: utf-8 -*-
"""Document processing utilities

Adapted from Adil Salhi's functions.py
https://gitlab.kaust.edu.sa/salhia/des_update_literature/blob/master/functions.py
"""
import xml.etree.ElementTree as ET
import json
from urllib.request import urlopen
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join
import os
import sys
import html
from collections import defaultdict
import numpy as np
from numpy import array as na
import datetime as dt
from collections import Counter
from pymongo import MongoClient
import pymongo
import bson

num_p= 26
default_retmax = 100000

############# utility #################
def valid_date(date_text):
	try:
		dt.datetime.strptime(date_text, '%Y/%m/%d')
	except:
		return False
	return True	

def get_url(s_date=None, e_date=None, pmids=None, retstart=0, retmax=default_retmax, summary=False): 
	if pmids:
		return f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmids}&retmode=xml&rettype=abstract'
	elif s_date and e_date and valid_date(s_date) and valid_date(e_date):
		if (summary):
			return 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&term=%22'+s_date+'%22[PDAT]%20:%20%22'+e_date+'%22[PDAT]'
		else:
			return 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retstart='+str(retstart)+'&retmax='+str(retmax)+'&term=%22'+s_date+'%22[PDAT]%20:%20%22'+e_date+'%22[PDAT]'
	else:
		raise ValueError('Failed to get url correctly!')

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def flatten(l_of_ls):
	result = []
	for l in l_of_ls:
		result.extend(l)
	return result

############# download #############################
def get_period_count(s_date, e_date):
	url = get_url(s_date=s_date, e_date=e_date, summary=True)
	return int(get_decoded_json(url)['esearchresult']['count'])

def get_year_count(year):
	s_date=str(year)+'/01/01'
	e_date=str(year)+'/12/31'
	return get_period_count(s_date, e_date)

def get_period_pmids(s_date, e_date, retmax=default_retmax):
	max_count = get_period_count(s_date, e_date)
	url_list = list(map(lambda x: get_url(retstart=x,s_date=s_date,e_date=e_date),list(range(0,max_count,retmax))))
	if(len(url_list) == 1):
		decoded_json = list(map(get_decoded_json, url_list))
	else:
		p = Pool(min(num_p, len(url_list)))
		decoded_json = list(p.map(get_decoded_json, url_list))
		p.close()
	pmid_list = flatten(list(map(lambda x: list(map(int,x['esearchresult']['idlist'])), decoded_json)))
	return set(pmid_list)

def get_year_pmids(year):
	s_date=str(year)+'/01/01'
	e_date=str(year)+'/12/31'
	return get_period_pmids(s_date, e_date)

def download_docs(pmids):
	return urlopen(get_url(pmids=pmids)).read()
	
######################### post download: parsing downloaded json, cleaning, covering missing  #########################
def get_downloaded_json_for_year(year):
        fname = 'json/'+str(year)+'/json_docs'
        if(not isfile(fname)):
                return []
        with open(fname) as f:
                json_text = f.read()
        split_text = json_text.split('"]["')
        merged_text = '","'.join(split_text)
        json_result = list(map(json.loads,json.loads(merged_text)))
        return json_result

def get_downloaded_pmids_for_year(year):
        json_docs = get_downloaded_json_for_year(year)
        pmid_list =list(map(lambda x: int(x['id']['pmid']), json_docs))
        return pmid_list

def get_downloaded_pmids_for_range(s_year, e_year):
        pmid_list = []
        for y in range(s_year, e_year+1):
                year_list = get_downloaded_pmids_for_year(y)
                pmid_list.extend(year_list)
        return pmid_list

def save_downloaded_pmids():
	downloaded_pmids = set(get_downloaded_pmids_for_range(1900,2017))
	arr_to_save = na(list(downloaded_pmids))
	with open(downloaded_fname, 'wb') as f:
		np.save(f, arr_to_save)
	return downloaded_pmids

def get_downloaded_pmids():
	fname = downloaded_fname
	if not isfile(fname):
		return save_downloaded_pmids()
	with open(fname, 'rb') as f:
		result = set(np.load(f))
	return result

def save_processed_pmids():
	existing = get_existing_pmids()
	downloaded = get_downloaded_pmids()
	result = existing.union(downloaded)
	arr_to_save = na(list(result))
	with open(processed_fname, 'wb') as f:
		np.save(f, arr_to_save)
	return result

def get_processed_pmids():
	fname = processed_fname
	if not isfile(fname):
		return save_processed_pmids()
	with open(fname, 'rb') as f:
		result = set(np.load(f))
	return result

#############################  automatic iterative updates ##########################
def save_last_update_date():
	today = dt.date.today()
	yesterday_str = dt.date(today.year, today.month, today.day - 1).strftime('%Y/%m/%d')
	with open(last_update_fname, 'wt') as f:
		f.write(yesterday_str)
	return

def get_last_update_date():
	with open(last_update_fname, 'rt') as f:
		last_update = f.readline().strip()
	return last_update

def get_downloaded_updates():
        fname = 'json/updates/json_docs'
        if(not isfile(fname)):
                return []
        with open(fname) as f:
                json_text = f.read()
        split_text = json_text.split('"]["')
        merged_text = '","'.join(split_text)
        json_result = list(map(json.loads,json.loads(merged_text)))
        pmid_list =list(map(lambda x: int(x['id']['pmid']), json_result))
        return pmid_list

def get_updates():
	s_date = get_last_update_date()
	e_date = '3000/12/31'
	updates = get_period_pmids(s_date,e_date)
	save_last_update_date()
	return updates

def process_update(pmid_list, num_p = num_p):
        pmid_lists = list(chunks(pmid_list, 100))
        pmid_lists = list(map(lambda x: str(x).strip('[]').replace(' ',''),pmid_lists))
        pmid_lists = list(chunks(pmid_lists, 10))
        except_counter = 0
        p=Pool(num_p)
        for l in pmid_lists:
                try:
                        print("downloading...")
                        doc_lists =p.map(download_docs,l)
                        print("parsing...")
                        doc_lists_parsed = p.map(parse_docs,doc_lists)
                        print("writing...")
                        directory = 'json/updates'
                        if not os.path.exists(directory):
                                os.makedirs(directory)
                        f=open(directory+'/json_docs', 'a+')
                        test = list(map(lambda x: json.dump(x,f),doc_lists_parsed))
                        f.close()
                except:
                        except_counter = except_counter + 1
                        continue
        p.close()
        print('exceptions:'+str(except_counter))


def process_duplicates(pmid_list, num_p = 4):
        pmid_lists = list(chunks(pmid_list, 100))
        pmid_lists = list(map(lambda x: str(x).strip('[]').replace(' ',''),pmid_lists))
        pmid_lists = list(chunks(pmid_lists, 10))
        except_counter = 0
        p=Pool(num_p)
        for l in pmid_lists:
                try:
                        print("downloading...")
                        doc_lists =map(download_docs,l)
                        print("parsing...")
                        doc_lists_parsed = p.map(parse_docs,doc_lists)
                        print("writing...")
                        directory = 'json/duplicates'
                        if not os.path.exists(directory):
                                os.makedirs(directory)
                        f=open(directory+'/json_docs', 'a+')
                        test = list(map(lambda x: json.dump(x,f),doc_lists_parsed))
                        f.close()
                except Exception as e:
                        except_counter = except_counter + 1
                        print('error: %s' % e)
                        continue
        p.close()
        print('exceptions:'+str(except_counter))

def parse_docs(xml_articles):
	super_root = ET.fromstring(xml_articles.decode('utf-8'))
	docs = []
	for root in super_root:
		for r in root.iter('MedlineCitation'):
			for s in r:
				if s.tag == 'PMID':
					pmid = s.text
				if s.tag == 'DateCreated':
					for u in s:
						if u.tag == 'Year':
							year = u.text
						if u.tag == 'Month':
							month = u.text
						if u.tag == 'Day':
							day = u.text
		article_date = year+'-'+month+'-'+day
		
		for r in root.iter('Journal'):
			for s in r:
				if s.tag == 'Title':
					journal_title = s.text
				
		for r in root.iter('ArticleTitle'):
			article_title = r.text
		
		abstract_text = ""
		for r in root.iter('AbstractText'):
			if r.text != None:
				abstract_text += r.text
				
		author_list = []
		for r in root.iter('Author'):
			author = ''
			for s in r:
				if s.tag in ['LastName', 'Initials', 'CollectiveName']:
					author += ' '+s.text
			if author != '':
				author_list.append(author.lstrip())
		
		docs.append(
            json.dumps(
                {
                    "meta" :
                    {
                        "src" : "pubmed",
                        "license" : "open-access",
                        "url" : "http://www.ncbi.nlm.nih.gov/pubmed/" + pmid
                    },
                "abstract" : abstract_text,
                "id" : {
                    "doi" : "",
                    "pmid" : pmid,
                    "ui" : "",
                    "pmcid" : ""
                    },
                "author" : author_list,
                "body" : "",
                "date" : article_date,
                "publication" : journal_title,
                "title" : article_title
                }
                )
            )
        
	return docs

############################## post processing: cleaning, removing duplicates, inserting into repo ######################
def get_duplicates_for_range(s_year, e_year):
	downloaded = get_downloaded_pmids_for_range(s_year,e_year)
	downloaded_freq = {k:v for k,v in Counter(downloaded).items()}
	duplicates = list(filter(lambda x: x[1]>1, downloaded_freq.items()))
	freq_duplicates = list(filter(lambda x: x[1]>2, downloaded_freq.items()))
	return duplicates, freq_duplicates

def get_duplicates(update=False):
	dups, _ = get_duplicates_for_range(1900,2017)
	duplicates = na(list(map(lambda x: x[0], dups)))
	if not update:
		return duplicates
	with open(duplicate_fname, 'wb') as f:
		np.save(f, duplicates)
	return duplicates

def get_max_id():
	with MongoClient() as conn:
		db = conn.biotextrepository
		max_id = db.articles.find( {}, { "_id": 1 } ).sort( "_id",  pymongo.DESCENDING ).next()['_id']
	return max_id

def add_id_(x):
	x[1]['_id'] = x[0]
	return x[1]

def add_ids(json_list):
	ids = list(map(bson.Int64,list(na([get_max_id()] * len(json_list)) + na(list(range(1, len(json_list)+1))))))
	return list(map(add_id_,list(zip(ids, json_list))))

#should be insert path really
def insert_year(year):
	downloaded = get_downloaded_json_for_year(year)
	to_insert = list(filter(lambda x: x['id']['pmid'] not in do_not_insert, downloaded))
	to_insert = add_ids(to_insert)
	print(len(to_insert))
	if len(to_insert) != 0:
		with MongoClient() as conn:
			db = conn.biotextrepository
			result = db.articles.insert_many(to_insert)
			return result.inserted_ids
	return []	

#for test removals (inserted_ids is returned by insert_year)
def delete_from_repo(id_list):
	with MongoClient() as conn:
		db = conn.biotextrepository
		db.articles.delete_many({'_id':{'$in':id_list}})
