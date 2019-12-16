# Visual Commonsense Reasoning

Explanation of the paper [Fusion of Detected Objects in Text for Visual Question Answering](http://arxiv.org/abs/1811.10830)

B2T2: Bounding Boxes in Text Transformer

## A brief description of the dataset

The dataset consists of more than 290k multiple choice QA problems with rationales derived from more than 110k movie clips. The task is - of stages answering and justification - in multiple choice setting. For an image, given a question, the model is provided 4 choices of answers, if it chooses the answer correctly, then it is provided four rationales and it must select the correct rationale. This setting is called _Q --> AR_ as for the model prediction to be correct requires both the chosen answer and then chosen rationale to be true. This task can be decomposed into two subtasks namely _Q --> A_ and _QA --> R_.

Following shows the task description:

![Task Description](images/task.png?raw=true)

Adversarial matching algorithm, which is based on maximum bipartite matching of queries and responses, selected on basis of relavance and similarity (using BERT and ESIM + ELMO, respectively) is used for creating the multiple choice dataset.


Checkout [this link](https://visualcommonsense.com/explore/) to explore the dataset and the results from the recognition to cognition networks.

## Section wise explanation of the paper

### 1. Introduction

This paper addresses the fact that the meaning of a word is systematically and predictably linked to the context in which it occurs. It shows that the right integration of visual and linguistic information can yield improvements in question answering. It addresses how to encode the visual and textual information into a neural architecture. Other interesting questions that it addresses but are yet to be discovered best solutions for : 

- How are text entities bound to objects seen in the images?
- Are text and images best integrated late (_late fusion_) or should the processing of one be conditioned on the analysis of the other (_early fusion_)? 
- How is cross-modal co-reference best encoded at all?
- Does it make sense to ground words in the visual world before encoding sentence semantics?

In the experiments shown in the paper it is found that :

1. Early fusion of co-references between textual tokens and visual features of objects was the most critical factor in obtaining improvements on VCR.
2. We found that the more visual object features we included inthe model's input, the better the model performed, even if they were not explicitly co-referent to the text, and that positional features of objects in the image  were  also  helpful.
3. We finally discovered that our models for VCR could be trained much more reliably when they were initialized from pre-training on Conceptual Captions a public dataset of about 3M images with captions

### 2. Problem Formulation

It is assumed that the data is comprised of _4-tuples_ :

1. _I_ is an image
2. _B = [b1, b2, ---,bm]_ is the list of bounding boxes referring to regions of _I_ where each _bi_ is identified by the lower left corner, height and width.
3. _T = [t1, ...,tn]_ is the passage of tokenized text, with the peculiarity that some of the tokens are not natural language but the explicit references to elements of _B_, and
4. _l_ is the binary label in {0,1}.

An image representation function *__&phi;__* that converts an image, perhaps after resizing andpadding,  to  a  fixed  size  vector  representation  of dimension _d_.
A pretrained textual representation capable of converting any tokenized passage of text into a vector of dimension _h_. Two types of such representations, one, without the context, *__E__* converts each token into a vector of dimension _h_ and a passage level representation *__&psi;__* returns the passage level vector representation of dimension _h_.

### 3. Models and Methods

Two types of architecture are proposed and discussed here. 

1. __Dual Encoder__:

The dual encoder model does not condition the extraction of features of any kind, textual or visual based on the analysis of the other. It basically takes the "dot prodct" of image features with the passage (text) features to find a single value and uses a sigmoid function to squash the results between zero and one. The target might be just to somehow implicitly fuse visual and textual information and back propagate end to end to achieve good results.

Class distribution is modeled as:

<center>_p(l = 1 | I, T)_ = 1 / (1 + e<sup>-&psi;(_E_(_T_))<sup>T</sup>_D_&phi;(_I_)</sup>)</center>

and the corresponding architecture is shown as:

![Dual Encoder](images/dualencoder.png?raw=true)


2. __B2T2__:

Class distribution is modeled as:

![B2T2 Distribution](images/b2t2distri.png?raw=true)

where a<sub>l</sub> &isin; R<sup>h</sup>, b<sub>l</sub> &isin; R, and are learned parameters. E'(I,B,R,T) is a non-contextualized representation for each token and of its position in text, but also of the content and position of the bounding boxes. The key difference from “Dual Encoder” is that text,  imageand bounding boxes are combined at the level of the non-contextualized token representations rather than right before the classification decision.

![B2T2 Distribution](images/b2t2et.png?raw=true)

The corresponding architecture of full B2T2 is shown here,

![B2T2 Diagram](images/earlyfusion.png?raw=true)

How are the embeddings calculated is shown in the following image

![B2T2 embeddings](images/embb.png?raw=true)

3. __Loss Funtion__: 

Binary crossentropy loss is used for label l.

4. __Pretraining of B2T2__:

Two tasks are used for pretraining 
	
	i) imposter identification
	ii) masked language model prediction

For the task of imposter identification, a random image is give a negative sampled caption and the model is asked to predict whether it is a suitable caption for the image and for the task of MLM, it is like in BERT where random tokens are masked and the model is asked to predict the tokens.

The pretraining of the model is done on the dataset __Conceptual Captions__.

![B2T2 pretraining](images/pretrain.png?raw=true)

