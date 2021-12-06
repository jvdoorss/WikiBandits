'''
Module defining Bandits: objects that are passed to Crawler and can:
        - pick an arm form the arms module
        - perform the arms action
        - reward the arm

The set of specialisations of the abstract Bandit is intended to contain a number of (2) arm
combiniations as well as a set of exploration methods to compare results and build intuition.
'''

import numpy as np

import arms

MAX_PAGES = 100

class Bandit:
    '''
    Abstract template for a multi-arm-evaluator

    For now serving as an explainatory placeholder, but would hold common
    features if the nr. of Bandits were to increase.
    '''
    def __init__(self,*bandit_arms,**kwargs):
        self.arms = bandit_arms
        self.current_arm = None
    def reward(self,url,subject,content,**metric):
        '''score result of the bandits action'''
    def pick_arm(self,url,subject):
        '''choose arm by some heuristic: e-greedy, ucb1,...'''
    def action(self,url,subject,log):
        '''perform the picked arms action'''
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
    '''
    Bandit with only one relevant arm, scoring positive or negative,
    pitched against the nothing option.
    Exploring the right arm is in fact useless here.
    '''
    def __init__(self,**kwargs):
        super().__init__(arms.Classifier(**kwargs),arms.LameArm())
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
    Bandit that picks new urls based on either a breadth-first (siblings)
    or depth-first (children) strategy depending on the existing linkage-properties
    of the respective pages and receives reward accordingly.
    '''
    def __init__(self,log,**kwargs):
        super().__init__(arms.Connector(log,arms.Pop.SIBLING,**kwargs),
                        arms.Connector(log,arms.Pop.CHILD,**kwargs))
        self.epsilon = kwargs.get('epsilon',0.1)
        self.current_arm = None
        self.log = log
    def reward(self,url,subject,content,**metric):
        return self.arms[self.current_arm].reward(url,subject,content,**metric)
    def pick_arm(self,url0,url1):
        '''epsilon-greedy approach'''
        if np.random.rand() > self.epsilon:
            return (0 if self.arms[0].estimate(url0,self.log.subject)
                            > self.arms[1].estimate(url1,self.log.subject)
                      else 1)
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


class Crampy(Bandit):
    '''
    Bandit with one linear arm and one discrete arm.

    It explores until one outperforms the other in the UCB way.
    '''
    def __init__(self,**kwargs):
        super().__init__(arms.Classifier(**kwargs),arms.LinearArm(**kwargs))
        self.winner = None
        self.current_arm = None
    def reward(self,url,subject,content,**metric):
        return self.arms[self.current_arm].reward(url,subject,content,**metric)
    def pick_arm(self,url,subject):
        '''explore first approach'''
        hoeffding = np.sqrt(2* MAX_PAGES / (sum(a.runs for a in self.arms)+1))
        if self.winner is None:
            if self.arms[0].mean - self.arms[1].mean > 4 * hoeffding:
                self.winner = 0
            elif self.arms[1].mean - self.arms[0].mean > 4 * hoeffding:
                self.winner = 1
        return np.random.randint(0,2) if self.winner is None else self.winner
    def action(self,url,subject,log):
        '''classifies the context-representation according the Evaluators current state'''
        self.current_arm = self.pick_arm(url,subject)
        arm = self.arms[self.current_arm]
        return arm.action(url,subject,log)

