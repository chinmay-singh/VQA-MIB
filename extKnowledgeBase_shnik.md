# Visual Question Answering Using External Knowledge Bases

*__Motivation__*: The task of VQA involves understanding of the contents in the images but often requires prior non visual information which can range from commonsense to encyclopedic information. For example to answer a question; "How many tyres does this truck have?" the model must understand what is meant by a "tyre" and that if there are two tyres on the left hand side of the truck there must be two more on the right hand side.

![tyre image](images/truck.jpg?raw=true)

This points out the weaknesses of the joint embedding approaches that they can only learn the knowledge present in the training set. An alternative to this is to decouple the reasoning _(neural network)_ from the actual storage. Structured representation of knowledge is important for the same. Knowledge bases like DBpedia, Freebase, NELL, ConceptNet etc., store the commonsense and factual information in a machine readable fashion. Each piece of information is stored as a tuple (or a graph) (arg1, rel, arg2) where arg1 and arg1 represent two concepts/ entities/ statements and rel represents the relationship betweeen them.

*__Methods__*: The methods concerning the use of external knowledge bases for visual question answering are presented in the following papers.

## Papers for Methods

### Explicit knowledge based reasoning for visual question answering

#### 1. Introduction

This section introduces the reader to the prolems with the existing approaches for visual question answering. There are mainly three kind of problems associated with the VQA systems that do not use external knowledge base for question answering. The typical approach of such kind uses a CNN to find the features from the images and an LSTM to find the textual features and finds a mapping from the question and image pairs to answers.

i) The method cannot explain how it arrived to its answer. This means that it is impossible to tell whether the model is answering based on the vissual information or because of the prevalenace of a particular answer in the training set.

ii) Because the model is trained on individual question answering pairs the range of the questions that can be answered accurately is limited.

iii) The LSTM approach is incapable of explicit reasoning except in very limited situations.

#### 2. Dataset Proposal and analysis

This paper proposes a dataset for VQA in external knowledge graphs. Questions to be asked about the image must be reducible to one of the proposed templates.

#### 3. VQA approach

__RDF Graph Construction__:

A information graph is constructed by using the features extracted from the image. Three types of visual concepts are detected by different cnn models including, objects, scenes and attributes. Visual concepts from the images are converted into RDF triplets and then into DBpedia entities with same semantic meanings.

__Answering Questions__: 

Quepy is used to convert the natural language questions in a format that can be used to parse the KB. Regular expressions are used to match the questions to one of the templates. Still slot phrases like "animal on the right" are in natural language.

__Mapping Slot Phrases to KB Entities__:

Phrases in slots *<obj>* are identified by provided locations/names/sizes and that in slots *<concept>* are mapped to DBpedia entities using predicate *wikiPageRedirects*

__Query Generation, Answer generation and reasoning__: Done using various predicates and templates. Basically kind of hard coded.

![evqa](images/evqa.png?raw=true)

### FVQA: Fact-Based Visual Question Answering

This paper basically uses the same approach as the previous one for a little bit different task of fact based vqa. The main difference in the approach here is that the template of the query is found using an lstm classifier instead of using Quepy. The LSTM used finds a softmax score over 32 template classes taken in account in the the problem formulatiion. The relations (triplets; arg1, rel, arg2) that are extracted using visual concepts and knowledge graphs are matched with the found template of query. The best matching is found on basis of similarity between question keywords and the args in relations. This matching is used to answer the question and provide the supporting fact.

![fvqa](images/fvqa.png?raw=true)


### Ask me anything: Free form visual question answering based on knowledge from external sources

#### 1. Introduction

This paper introduces another approach for visual question answering based on integration of visual and textual features along with querying the knowledge base. It presents a late fusion technique to do so. The technique is shown and explained in the figure below. Next section moves into the details of the network architecture.

![external vqa joint embedding approach](images/vqaext.png?raw=true)

#### 2. Related works

#### 3. Extracting, Encoding and Merging

i. __Attribute-based Image Representation__:
	
This is formulated as a problem of multi-label classification over a vocabulary of words collected from the captions occuring in MS-COCO dataset, which seems to be a reasonable thing to do. Now, to collect features from the image to carry out the classification, region proposals are used rather than directly using the features because some of the attributes might directly be a consequence of a particular part of an image.

Multiscale Combinatorial Grouping is used for generating proposals. For each proposal, a VGG network pretrained on Imagenet is used to find the features (fine tuning on MS-COCO image-attributes dataset) and a c-way softmax which produces the probability distributions over c vocabulary attributes (c = 256 in this case) is used. Finally a cross-hypothesis max-pooling is used to integrate the outputs into a single prediction vector V<sub>att</sub>(I).

ii. __Caption-based Image Representation__:

Caption-LSTM is used to generate the captions for the given image. Higher level attributes like that extracted in attribute based image representations are used to generate the caption for the image. Five different captions are created using beam search, constituting the internal textual representation of the image. The last hidden state of the LSTM for all 5 generated captions are average pooled into a single prediction vector V<sub>cap</sub>(I).

iii. __Relating to the knowledge base__:

The top-5 extracted attribute are queried in the knowledge base DBpedia and a document is created from the summaries (the textual description of th e attributes). The document is then passed through the DOC2VEC model to create a fixed length vector for this information. This vector is termed a V<sub>know</sub>(I).

#### 4. A VQA Model with Multiple Inputs

A LSTM sequence to sequence encoder-decoder model is used to encode the information in the question and generate the corresponding answer. Parameters are shared between encoder and the decoder.

__Working__: 

Initialize the hidden state with 

![Hidden state initial](images/vqaeqn.png?raw=true)

carry out the encoding with the parameters to be trained end-to-end. In the decoder stage the hidden state matrix is used to find the probability distribution over the vocabulary. The following loss function is used to carry out the training. And batches of 100 image QA pairs are used for training of this LSTM.

![loss function](images/vqaloss.png?raw=true)

