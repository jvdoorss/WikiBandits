
'''
Module to define Arms, i.e. object with
        - a statistical etimator of a reward
        - an action with some captured result
        - a true reward calculated on the result
          and an optimising method to give the estimator feedback

In this case the action always involves downloading a page, the estimator acts on prior information,
like the url, and the reward acts on the downloaded content.

The arms can be fed to Bandits in the bandits module.
'''

from enum import Enum

import torch
from torch.nn import Sequential, Linear, CrossEntropyLoss, Softmax
from scipy.special import ndtr

from crawler import PropsFetchError, wiki_relevance, wiki_size, MAX_SIZE

class Arm:
    '''
    Abstract template for a contextual bandit arm that downloads web pages

    Serves mainly as a placeholder for now.
    '''
    def __init__(self,**parameters):
        self._parameters = parameters
    def estimate(self,url,subject):
        '''
        Predictor to map the state of the Arm and the context-variables
        to a prediction of the reward
        '''
    def action(self,url,subject,log):
        '''
        Download and additional operation to be performed after the arm is picked,
        but before its reward is calculated
        '''
    def reward(self,url,subject,content,**metric):
        '''
        Function to:
         - calculate the true reward from the Arm's response (the downloaded content)
         - update the Arm's state from this feedback
        '''

class Classifier(Arm):
    '''
    Class to evaluate relevance of a <url,subject,size> context using logistic regression
    from a pretrained word2vec model.
    '''
    def __init__(self,**parameters):
        '''
        @param max_size: size in bytes
        @param lr: learning rate fro SGD
        '''
        super().__init__(**parameters)
        self.classifier = Sequential(Linear(2,2),Softmax(dim=-1))
        self.loss = CrossEntropyLoss()
    def _evaluate(self,url,subject):
        '''
        Helper function to create an intermediate representation
        for both the context and its estimated effect
        '''
        context_representation = torch.Tensor([ wiki_relevance(url,subject),
                                    wiki_size(url,self._parameters.get('max_size',MAX_SIZE))])
        return self.classifier(context_representation)
    def estimate(self,url,subject):
        '''estimate of the reward based on the internal state'''
        res = self._evaluate(url,subject)
        return (res[1] - res[0]).item()
    def action(self,url,subject,log):
        estimate =  self.estimate(url,subject)
        log.add_url(url,True,estimate = estimate)
        return estimate
    def _response(self,subject,content,size):
        '''
        Custom translation of the resulting content to a reward.

        A more accurate implementaion could involve:
        - hyper-linkedness to other downloaded pages
        - time a user spends on the page
        - general click-count for the page (if this is public info)
        - stats like nr.of contributors, last update, references,...
        - separate size-weight for text and images
        '''
        return content.count(subject) / 100 - size / self._parameters.get('max_size',MAX_SIZE)
    def reward(self,url,subject,content,**metric):
        # get the response and adjest the learning rate accordingly,
        # this way the binary classifier takes the size into account
        response = self._response(subject,content,metric.get('size',0))
        lr = self._parameters.get('lr',0.1) * abs(response)

        # train logistic regressor as usual
        self.classifier.train()
        prediction = self._evaluate(url,subject)
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=lr)
        loss = self.loss(prediction[None],torch.LongTensor([1 if  response > 0 else 0]))
        self.classifier.zero_grad()
        loss.backward()
        optimizer.step()
        self.classifier.eval()

        return response

class LameArm(Arm):
    ''' Arm performing no download, returning the 0 (neutral) reward'''
    def estimate(self,url,subject):
        return 0
    def action(self,url,subject,log):
        '''Do add url with False-flag, to not visit again. TODO: reconsider this'''
        log.add_url(url,False)
        return 0
    def reward(self,url,subject,content,**metric):
        return 0

class Pop(Enum): QUEUE, SIBLING, CHILD = range(3)

class Connector(Classifier):
    '''
    Arm like Classifier but using the linkage i.s.o. the word2vector as url-score.
    '''
    def __init__(self,log,mode,**parameters):
        super().__init__(**parameters)
        self.log = log
        self.mode = mode
    def _evaluate(self,url,subject):
        '''
        Helper function to create an intermediate representation
        for both the context and its estimated effect
        '''
        mean,std = self.log.nodes.linkage.mean(),self.log.nodes.linkage.std()
        linkage_score = ndtr((self.log.linkage(url) - mean)/std)
        size_score = 0
        try:
            size_score = wiki_size(url,self._parameters.get('max_size',MAX_SIZE))
        except PropsFetchError:
            pass
        context_representation = torch.Tensor([ linkage_score, size_score])
        return self.classifier(context_representation)
