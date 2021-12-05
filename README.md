# Tool to collect subject documentation from wikipedia

Exercise on Multi-Armed Contextual Bandits. 

**Crawler** selects and weblinks and contains a **Bandit** that estimates their validity using one of a number of binary classifiers, its **Arms**. A number of the later objects are implemented to test different possible scenarios. 

An arm-action results in a download which reveals the cost/reward after the facts, and only for the download chosen. In a true multi-armed approach, each arm could have a separate reward, but should also involve a separate download. When merely comparing estimators of different rewards for the same page, all rewards can be known across arms. The included **Bandits** are intended to explore both varying estimators for a single url as to pith urls against each other.

Modules: numpy, scypi, torch, pandas, requests, bs4 (beautifulsoup) and wikipedia2vec. Using a pretrained version of the latter, to be [downloaded](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.pkl.bz2) separately.

## TODO:

- update/add reward function
- test convergence in a long run
- test other exploration methods 