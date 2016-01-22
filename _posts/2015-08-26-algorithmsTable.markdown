---
layout: post
title:  " Summary of Mathematics in Machine Learning Algorithms"
author: Hetian Chen
date:   Aug 26, 2015
excerpt: This post covers a partial list of parametric machine learning algorithms.
categories: machine-learning
---

This post covers a partial list of parametric machine learning algorithms.  
### Generative models
#### Naive Bayes
* input/output  
	$\left\lbrace(x_{n},y_{n})\right\rbrace ^{N}_{n=1}$  
	$y \in \left\lbrace 0,1,...C \right\rbrace$ 
* model  
	$P(\boldsymbol {X=x}, Y = y)= P(Y=y)P(\boldsymbol {X=x}\vert Y=y)$  
* likelihood  
	$P(D)=\prod\limits_{n=1}^{N} P(\boldsymbol {X_{n} = x_{n}}, Y_{n}= y_{n})$  
* train $\boldsymbol {w}$  
	maximize $logP(D)$
* solve $\boldsymbol {w}$  
	$P(Y=y) \rightarrow Y \sim Multinomial(N,p)$  
	$P(\boldsymbol {x_{k}}\vert Y=y) \rightarrow \boldsymbol {x_{k}}\vert Y=y \sim Binomial (N,p^{*})$  
* decision  
	$y^{*} = \arg\max_{c\in C} P(y=c\vert \boldsymbol {x})=\arg\max_{c\in C} P(y=c)P(\boldsymbol {x} \vert y=c)$  

### Discriminative models       
#### Logistic Regression (binary outcome)
* input/output  
	$\left\lbrace(x_{n},y_{n})\right\rbrace ^{N}_{n=1}$  
	$y \in \left\lbrace 0,1 \right\rbrace$  
* model  
	$P(y=1\vert \boldsymbol{x,w}) = \sigma (\boldsymbol{w^{T}x})$  
* likelihood  
	$P(D)=\prod\limits_{n=1}^{N} [P(y_{n}=1 \vert \boldsymbol {x_{n},w})]^{y_{n}}[1-P(y_{n}=1 \vert \boldsymbol {x_{n},w})]^{1-y_{n}}$  
* error measure  
	$ \text{cross entropy: }-logP(D)= -\sum_{n=1}^N y_{n}log\sigma(\boldsymbol {w^{T}x_{n}}) + (1-y_{n})log(1-\sigma(\boldsymbol {w^{T}x_{n}}))$  
* train $\boldsymbol {w}$  
	minimize $-logP(D)$
* solve $\boldsymbol {w}$  
numeric method:  

	* gradient descent: $\boldsymbol {w^{t+1}} = \boldsymbol {w^{(t)}} - \alpha \frac {\partial {-logP(D)}} {\partial {\boldsymbol {w^{(t)}}}}$  
	* Newton's method: $ \boldsymbol {w^{t+1}} = \boldsymbol {w^{(t)}} - [H^{(t)}]^{-1} \frac {\partial {-logP(D)}} {\partial {\boldsymbol {w}^{(t)}}}$  

* decision  
	$y^{*} = \arg\max_{c\in \left\lbrace 0,1 \right\rbrace } P(y=c \vert \boldsymbol {x,w},b)$  

#### Multinomial Logistic Regression (multi-class outcome)
* input/output  
	$\left\lbrace(x_{n},y_{n})\right\rbrace ^{N}_{n=1}$  
	$y \in \left\lbrace 0,1,...C \right\rbrace$   
* model  
	$P(C_{k} \vert \boldsymbol {x}) = \frac {e^{\boldsymbol {w}^T_k}\boldsymbol {x}} {\sum_{j \in \left\lbrace 0,1,...C \right\rbrace} e^{\boldsymbol {w}^T_j \boldsymbol {x}}}$  
* likelihood  
	$\prod\limits_{n=1}^{N} P(y_{n} \vert \boldsymbol {x_{n}})= \prod\limits_{n=1}^{N} \prod\limits_{k=1}^{K} P(C_{k} \vert \boldsymbol {x_{n}})^{y_{nk}}$  
	$
	y_{nk}= \begin{cases}
	1, \text {if }y_{n}=k\\\\
	0, otherwise
	\end{cases}
	$  
* error measure  
	$-logP(D)$  
* train $\boldsymbol {w}$  
	minimize cross entropy (maximize log likelihood) as above  
* solve $\boldsymbol {w}$  
	same as above  
* decision  
	$y^{*} = \arg\max_{k} P(y=C_{k} \vert \boldsymbol {x})$  

#### Linear Regression 
* input/output  
	$\left\lbrace(x_{n},y_{n})\right\rbrace ^{N}_{n=1}$  
	$y \in R $   
* model  
	$y\vert \boldsymbol {x} \sim N(\boldsymbol {w^{T}x},\sigma ^{2})$  
	$P(y\vert \boldsymbol {w,x}) = \frac {1} {\sqrt{2\pi}\sigma} exp(-\frac {(y-\boldsymbol {w^{T}x})^{2}} {2\sigma ^{2}})$  
* likelihood  
	$\prod\limits_{n=1}^{N} P(y_{n} \vert \boldsymbol {x_{n}})$
* error measure  
	residual sum of square (RSS): $\sum_{n=1}^{N} (y_{n}-\boldsymbol {w^{T}x_{n}})^{2}$  
* train $\boldsymbol {w}$  
	minimize RSS = maximize log likelihood
* solve $\boldsymbol {w}$  
	largrange multiplier
* decision  
	$y^{*} = \boldsymbol {w^{T}x}$  
	Note: bias term already absorbed in $\boldsymbol {w}$  

#### Decision Tree
* input/output  
	$\left\lbrace(x_{n},y_{n})\right\rbrace ^{N}_{n=1}$  
	$y \in \left\lbrace 0,1,...C \right\rbrace$ 
* model  
	$\text{Information Gain} = H\left[Y \right] - H\left[Y \vert X \right]$  
* train model  
	maximize Information Gain = minimize Conditional Entropy  
* decide which attribute to split next  
	$\arg\min_{k\in K} H(Y \vert X_{k})$  
	$H(Y \vert X) = \sum_{j} P(X=v_{j})H(Y \vert X = v_{j})$  
	$H(Y \vert X=v_{j}) = -\sum_{k=1}^{m} P(Y=k \vert X=v_{j})logP(Y=k \vert X= v_{j})$  
* decision  
	follow the decision tree built above

#### Perceptron
* input/output  
	$\left\lbrace(x_{n},y_{n})\right\rbrace ^{N}_{n=1}$  
	$y \in \left\lbrace -1,1 \right\rbrace$  
* model  
	$y_{n} = sign(\boldsymbol {w^{T}x_{n}})$
* error measure  
	0/1 loss function:
		$\varepsilon = \sum_{n} I\left[y_{n} \neq sign(\boldsymbol {w^{T}x_{n}}) \right]$
* train $\boldsymbol {w}$  
	minimize 0/1 loss function
* solve $\boldsymbol {w}$  
	$\boldsymbol {w}^{(t+1)} \leftarrow \boldsymbol {w}^{(t)} + y_{n}\boldsymbol {x_{n}}$  
	update if $y_{n+1} \neq sign(\boldsymbol {w^{T}x_{n+1}})$
* decision  
	$y^{*} = \boldsymbol {w^{T}x}$

#### Suppoart Vector Machines
* input/output  
	$\left\lbrace(x_{n},y_{n})\right\rbrace ^{N}_{n=1}$  
	$y \in \left\lbrace -1,1 \right\rbrace$  
* model  
	$h(\boldsymbol {x}) = sign(\boldsymbol {w^{T}\varphi(x)} + b)$
* error measure  
	hinge loss:
		$l^{hinge}(f(\boldsymbol{x}),y) = \begin{cases}
0, yf(\boldsymbol{x})\geq 1\\\\  
1- yf(\boldsymbol{x}), otherwise
\end{cases}
$  
* train $\boldsymbol {w}$  
	minimize hinge loss:
		$\min_{\boldsymbol {w},b} C\sum_{n} \varepsilon_{n}+\frac {1} {2} \|\boldsymbol{w}\|^{2}$  
	subject to $\varepsilon_{n} = max(0,1-yf(\boldsymbol {x})), \varepsilon \geq 0$  
* solve $\boldsymbol {w}$  
	quadratic programming
* decision  
	$y^{*} = \boldsymbol {w^{T}\varphi(x)} + b$






