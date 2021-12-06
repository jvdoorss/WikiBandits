'''
This module provides tools to gather content surrounding a single subject on wikipedia efficiently.

Essential classes are:
    - the SubjectLog, that keeps a structured history of aquired knowledge, adding
      to the context for the Arms and Bandits in the respective modules.
    - the Crawler, random walking trough links in the (increasing) SubjectLog based
      on a given Bandit's decisions.

To use the Wikipedia2Vec functionality, enwiki_20180420_100d.pkl is needed:
http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.pkl.bz2
'''

import os
from contextlib import contextmanager

import requests
from wikipedia2vec import Wikipedia2Vec
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

MAX_SIZE = 1000000

#######################################################################################
#
# Utilities to handle and store wikipedia pages
#

def urlkey(url):
    '''converts url to a unique, shorter identifier'''
    return url.split('/')[-1].replace('_',' ')

def complete_url(url):
    '''extends the short urls'''
    prefix = 'https://en.wikipedia.org'
    sections = url.strip('/').split('/')
    if sections[0] == 'wiki':
        return '/'.join([prefix]+sections)
    return url

@contextmanager
def download(url,subject):
    '''
    Contextmanager to download web-page contents and save it to an appropriate folder afterwards

    @param url: page to be downloaded
    @param subject: reference to the repo to which to save, create if not exists.
    '''
    folder = 'repos/'+subject+'/'
    name = url.split('/')[-1]+'.html'
    path = folder + name
    page = requests.get(url)
    try:
        os.makedirs(folder,exist_ok=True)
        yield page,path
        with open(path, "x",encoding="utf8") as file:
            file.write(page.text)
    except FileExistsError:
        pass

class SubjectLog:
    '''
    Log for (non-)downloaded pages.

    Adding a page as node to the log with the download-flag on will 'explore' the page and
    hence perform the download and store the page to the repo.

    @attrib subject: the subject at hand
    @attrib nodes: the pages that were considered an possibly downloaded
    @attrib edges: outging hyperlinks from all downloaded pages
        TODO:   This could be used to set up a measure for subsequent url's,
                and perhaps replace wikipedia2vec
                unused for now
    @attrib unexplored: queue of all outgoing links that were encountered, possibly many duplicates,
                        which are skipped when iterating with 'pop()'

    '''
    def __init__(self,subject):
        self.subject = subject
        self.edges =  pd.DataFrame(columns = ['source','target'])
        self.nodes = pd.DataFrame(columns = ['url','size','linkage','estimate','reward'])
        self.unexplored = []
    def download(self,url,**url_props):
        '''
        Performs the download. The download context will add the page to the repo afterwards,
        if no exceptions are raised.
        '''
        try:
            with download(url,self.subject) as (page,path):
                soup = BeautifulSoup(page.text,features='html.parser')
                links = [{  'source':url,
                            'target': href}
                            for link in soup.find_all('a')
                            if isinstance(href := link.get('href'),str) and '/wiki/' in href]
                node = pd.Series({**{   'path':path,
                                        'url':url,
                                        'size':int(page.headers['content-length']),
                                        'linkage' : len(links)},
                                    **url_props},
                                    name = urlkey(url))
                return node,links
        except:
            # a faulty link gets added as empty row
            node = pd.Series(name = urlkey(url),dtype=object)
            links = []
            return node,links
    def pop(self):
        '''
        Pops urls from the front of the unexplored queue until a new one
        (not yet in nodes) is found
        '''
        if not self.unexplored:
            raise EmptyQueue()
        current,*self.unexplored = self.unexplored
        while urlkey(current) in self.nodes.index:
            if not self.unexplored:
                raise EmptyQueue()
            current, *self.unexplored = self.unexplored
        return complete_url(current)
    def add_url(self,url,download_flag,**url_props):
        '''Add url to log and download to repo ifneedsbe'''
        if download_flag:
            node,links = self.download(url,**url_props)
            self.edges = self.edges.append(links,ignore_index=True)
            self.unexplored += [link['target'] for link in links]
        else:
            node = pd.Series({**{'url':url},**url_props},name = urlkey(url))
        self.nodes = self.nodes.append(node)
        return self
    def length(self):
        '''total nr of urls explored: downloaded or not'''
        return len(self.nodes)
    def size(self):
        '''total downloaded size'''
        return self.nodes['size'].sum()
    def downloads(self):
        '''total nr of downloaded urls'''
        return len(self.nodes.path.dropna())
    def plot(self,max_size = MAX_SIZE):
        '''
        Plots metrics of the bandit-actions.
        TODO: add chosen arm (!), plot cumulative to show total regret
        '''
        self.nodes[['estimate','reward','size']]\
            .assign(size = lambda df: df['size']/max_size)\
            .plot()
    def linkage(self,url):
        '''returns nr of logged incoming links'''
        return len(self.edges[self.edges.target.apply(urlkey) == urlkey(url)])
    def pick_sibling(self,url):
        '''
        Returns most connected sibling of url in the already built-up graph

        Note:   Initially this will return all wikipedia system pages, recording them as node and
                ignoring them in the long run.
        '''
        parents = self.edges[self.edges.target.apply(urlkey) == urlkey(url)][['source']]\
                        .drop_duplicates()
        siblings = self.edges.merge(parents,on = 'source',how='right')[['target']].drop_duplicates()
        incoming = self.edges.merge(siblings,on = 'target',how='inner')\
            .groupby('target')\
            .count()\
            .sort_values(by='source',ascending=False)\
            .assign(url = lambda df : df.index.map(complete_url))
        incoming.index = incoming.index.map(urlkey)
        incoming.drop(self.nodes.index.map(urlkey),errors='ignore',inplace=True)
        return incoming.url[0]
    def pick_child(self,url):
        '''returns most connected child of url according to current graph'''
        children = self.edges[self.edges.source.apply(urlkey) == urlkey(url)][['target']]\
                        .drop_duplicates()
        incoming = self.edges.merge(children,on = 'target',how='inner')\
            .groupby('target')\
            .count()\
            .sort_values(by='source',ascending=False)\
            .assign(url = lambda df : df.index.map(complete_url))
        incoming.index = incoming.index.map(urlkey)
        incoming.drop(self.nodes.index.map(urlkey),errors='ignore',inplace=True)
        return incoming.url[0]

class PropsFetchError(Exception):
    '''Blanket solution when fetching page data fails.'''

def wiki_size(url,max_size):
    '''returns the relative size of the page from header info'''
    try:
        props = requests.head(url).headers
        return int(props['content-length']) / max_size
    except:
        raise PropsFetchError()
def wiki_relevance(url,subject):
    '''evaluates the <url,subject> correlation within wikipedia2vec'''
    key_word = url.split('/')[-1].replace('_',' ')
    try:
        key_vec = np.array(wiki_relevance.embedding.get_entity_vector(key_word))
        word_vec = np.array(wiki_relevance.embedding.get_word_vector(subject))
    except:
        raise PropsFetchError()
    return (key_vec @ word_vec) / np.linalg.norm(key_vec) / np.linalg.norm(word_vec)
wiki_relevance.embedding = Wikipedia2Vec.load('enwiki_20180420_100d.pkl')

#######################################################################################
#
# Tool for iterating pages

class EmptyQueue(Exception):
    '''Signals no more unexplored url's (shouldn't happen)'''

class Crawler:
    '''
    Tool to iterate pages for a given subject using a specified bandit-evaluator
    '''
    def __init__(self, url0, subject,log,bandit):
        self.current_url = url0
        self.subject = subject
        self.log = log.add_url(url0,True)
        self.bandit = bandit
    def pop(self):
        '''Pops next candidate from front of the queue'''
        self.current_url = self.bandit.pop(self.log)
        return self
    def process(self):
        ''' Perform bandit action and rewarding'''
        self.bandit.action(self.current_url,self.subject,self.log)

        key = urlkey(self.current_url)
        if (path := self.log.nodes.at[key,'path']) and pd.notna(path):
            with open(path,'r',encoding="utf8") as file:
                content = file.readlines()
            size = self.log.nodes.at[key,'size']
            res = self.bandit.reward(self.current_url,self.subject,content,size = size)
            self.log.nodes.at[key,'reward'] = res
        return self
    def done(self,max_pages,max_size):
        '''Signal ending criterium'''
        print('downloaded ',self.log.size(),' of ',max_size)
        return self.log.length() >= max_pages or self.log.size() >= max_size
