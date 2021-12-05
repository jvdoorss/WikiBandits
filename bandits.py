from enum import Enum

import numpy as np

import arms

class Bandit:
    '''
    Abstract template for a multi-arm-evaluator

    For now serving as an explainatory placeholder, but would hold common features if
    the nr. of Bandits were to increase.
    '''
    def __init__(self,*arms,**kwargs):
        self.arms = arms
        self.current_arm = None
    def reward(self,url,subject,content,**metric): pass
    def pick_arm(self,url,subject):pass
    def action(self,url,subject,log): pass
    def pop(self,log):
        '''redefining pop to enable the bandit to overwrite it'''
        return log.pop()

class ClassyBandit(Bandit):
    '''
    Wrapper for the classifier, interpreting it as a one-armed bandit
    '''
    def __init__(self,**kwargs):
        super().__init__(arms.Classifier(**kwargs),**kwargs)
    def pick_arm(self,url,subject):
        return 0
    def action(self,url,subject,log):
        return self.arms[0].action(url,subject,log)
    def reward(self,url,subject,content,size):
        return self.arms[0].reward(url,subject,content,size = size)

class Lefty(Bandit):
    def __init__(self,**kwargs):
        self.arms = [arms.Classifier(**kwargs),arms.LameArm()]
        self.epsilon = kwargs.get('epsilon',0.1)
        self.current_arm = None
    def reward(self,url,subject,content,**metric):
        return self.arms[self.current_arm].reward(url,subject,content,**metric)
    def pick_arm(self,url,subject):
        '''epsilon-greedy approach'''
        if np.random.rand() > self.epsilon:
            return 0 if self.arms[0].estimate(url,subject) > 0 else 1
        return np.random.randint(0,1)
    def action(self,url,subject,log):
        '''classifies the context-representation according the Evaluators current state'''
        self.current_arm = self.pick_arm(url,subject)
        arm = self.arms[self.current_arm]
        return arm.action(url,subject,log)

class Genealogist(Bandit):
    '''
    Bandit that picks new urls based on either a breadth-first (siblings) of depth-first (children)
    strategy depending on the existing linkage-properties of the respective pages and receives reward accordingly.


    '''
    def __init__(self,log,**kwargs):
        self.arms = [   arms.Connector(log,arms.Pop.Sibling,**kwargs),
                        arms.Connector(log,arms.Pop.Child,**kwargs)]
        self.epsilon = kwargs.get('epsilon',0.1)
        self.current_arm = None
        self.log = log
    def reward(self,url,subject,content,**metric):
        return self.arms[self.current_arm].reward(url,subject,content,**metric)
    def pick_arm(self,url0,url1):
        '''epsilon-greedy approach'''
        if np.random.rand() > self.epsilon:
            return 0 if self.arms[0].estimate(url0,self.log.subject) > self.arms[1].estimate(url1,self.log.subject) else 1
        return np.random.randint(0,1)
    def action(self,url,subject,log):
        '''classifies the context-representation according the Evaluators current state'''
        arm = self.arms[self.current_arm]
        return arm.action(url,subject,log)
    def pop(self,log):
        current_url = log.nodes.url.dropna().iloc[-1]
        url0 = log.pick_sibling(current_url)
        url1 = log.pick_child(current_url)
        self.current_arm = self.pick_arm(url0,url1)
        return url1 if self.current_arm else url0
