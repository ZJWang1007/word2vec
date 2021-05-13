Word2Vec Implementation With Numpy
====

First thing's first, this can never be a tutorial nor an example code. It's just a personal attempt to implement Word2Vec. Bugs and low efficiencies can be found everywhere in my code so both your advice and opinion are well welcomed here. 

References 
----
- The paper that started it all:

https://arxiv.org/pdf/1301.3781.pdf

Mikolov T, Chen K, Corrado G, et al. Efficient estimation of word representations in vector space[J]. arXiv preprint arXiv:1301.3781, 2013.

- The paper that gives the best explanation:
  
https://arxiv.org/pdf/1411.2738.pdf 

Rong X. word2vec parameter learning explained[J]. arXiv preprint arXiv:1411.2738, 2014. 

Environment
----
I'm using **Numpy**, **jieba**(for word segmentation), **re**(regular expression for leaving out all the non-Chinese characters) and **zhconv**(for turning traditional Chinese into simplified Chinese if needed). 

