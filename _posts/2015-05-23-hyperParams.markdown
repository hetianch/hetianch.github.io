---
layout: post
title:  "Summary of Automatic Hyperparameter Selection Packages"
author: Hetian Chen
date:   May 23, 2015
excerpt: Grid search is the most intuitive way of performing hyperparameter optimization. However, it suffers from the curse of dimensionality becuase the number of joint values grows exponentially with the number of hyperparameters. Here is a list of packages, which use better methods of hyperparameter optimization.
categories: machine-learning
---
In the task of predictive modeling, often times, we'd like to compare the performance of algorithms. Beside selecting the appropriate metrics, another important thing is **to tune the hyperparameters** for each algorithm to make a fair comparison. Grid search is the most intuitive way of performing hyperparameter optimization. However, it suffers from the curse of dimensionality becuase the number of joint values grows exponentially with the number of hyperparameters. Here is a list of packages, which use better methods of hyperparameter optimization.

|Name | Language | Method | Paper | Comment|
|-----|----------|--------|-------|--------|
|Spearmint|Python|Gaussian processes|Snoek, 2012|continuous hyper-parameters|  
|BayesOpt|C++ with Python and Matlab/Octave interfaces|Gaussian processes|Martinez-Cantin, 2014|continuous hyper-parameters|  
|hyperopt|Python|Tree-structured Parzen Estimator|Bergstra, 2013|included in autoweka|  
|SMAC|Java|sequential model-based algorithm configuration|Hutter, 2010|included in autoweka|  
|REMBO|Matlab|random embedding bayesian optimization|Wang, 2013|matlab,slow|  
|MOE|C++/Python|Bayesian global optimization|Clark, 2014|flexible, heavy customization|
|AutoWeka|JAVA/Python wrapper|TPE/SMAC|Thornton, 2013|overall convenient|  



References:  

* Snoek, Jasper, Hugo Larochelle, and Ryan P. Adams. "Practical Bayesian optimization of machine learning algorithms." Advances in neural information processing systems. 2012.  

* Martinez-Cantin, Ruben. "BayesOpt: a Bayesian optimization library for nonlinear optimization, experimental design and bandits." The Journal of Machine Learning Research 15.1 (2014): 3735-3739.  

* Bergstra, James, Dan Yamins, and David D. Cox. "Hyperopt: A Python library for optimizing the hyperparameters of machine learning algorithms." Proceedings of the 12th Python in Science Conference. 2013.  

* Hutter, Frank, et al. "Time-bounded sequential parameter optimization." Learning and intelligent optimization. Springer Berlin Heidelberg, 2010. 281-298.  

* Wang, Ziyu, et al. "Bayesian optimization in a billion dimensions via random embeddings." arXiv preprint arXiv:1301.1942 (2013).
APA  

* Scott Clark. Metric Optimization Engine. A global, black box optimization engine for real world metric optimization. (2014 by yelp)  

* Thornton, Chris, et al. "Auto-WEKA: Combined selection and hyperparameter optimization of classification algorithms." Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2013.  








