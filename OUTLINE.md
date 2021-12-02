# ai-coding-challenge

## Description
The programming challenge consists in building a contextual multi-armed bandit (CMAB) agent for downloading Wikipedia webpages related to a given an input subject and considering limited webpage storage.
Conceptually, the agent will be composed of at least two sub-systems:
 - a webpage crawler that explores the Wikipedia links (uri's)
 - a CMAB with two actions (or, arms) that chooses to either download a page or not based on the present context. The context representation is up to the candidate's choice (the uri's parts for example).

## Requirements
 - input for the algorithm: start uri, subject (this can be a word or phrase describing what to be downloaded) and a size budget for downloaded data (in KiB/MiB, etc)
 - output: downloaded webpages and some information/measurement/plot that shows that the agent performs as expected
 - programming language: [Julia](https://julialang.org) or [Python](https://python.org)
 - the code will be made accessible to VUB through a GitHub link (just reply with the link in an email)
 - time to deliver: 3 days and nights :) (assuming you receive the challenge on Monday morning, have it done by Thursday morning) 

## Challenging parts
A particular challenging aspect of the program is that you are left with quite a few choices in the implementation! A non-exhaustive list of challenging aspects:
 - the type of CMAB to use
 - the representation of the context
 - the reward functions for the two arms of the bandit
 - performance evaluation; i.e., showing that the algorithm(s) work as expected (mostly for the CMAB)
 - the balance between correct functionality-complexity and simplicity-readability of the code

## Tips
 - document the state of functionality that the code is in, what challenges were encountered, what works and what not
 - try to make it work first (i.e., choose a simple, discrete context representation; for example fixed number of values between [-1 and 1] with 1 indicating (cosine) similarity between uri representation and subject representation)
 - use simple rewards at first; i.e., constant for skipping a page, some number between [0, 1], with 1 indicating that the page is a good fit wrt. the input subject
 - try several CMABs if time allows
 - write high quality compact, easy to understand code (PEP8 it, lint it if Python, use a good style for Julia)
 - document the code and usage (it does not have to be extensive)
 - test it (optional)
 - CI it (optional)
 - Please note that we're evaluating your thought process, technical proficiency and problem-solving skills (in a complex task) and not going to use the system in production ;) Hence, we're not looking for perfection but do expect some good effort and the ability to explain the choices made.

## Documentation
Quite a bit of documentation can be found online, underneath are just a couple of starting points...
 - [Nice paper on bandits](https://arxiv.org/pdf/1904.07272.pdf)
 - [An article with sample Python implementations](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html)
 - [Kaggle notebook on CMABs](https://www.kaggle.com/phamvanvung/cb-linucb)
