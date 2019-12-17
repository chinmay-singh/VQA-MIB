# Visual Question Answering Using External Knowledge Bases

*__Motivation__*: The task of VQA involves understanding of the contents in the images but often requires prior non visual information which can range from commonsense to encyclopedic information. For example to answer a question; "How many tyres does this truck have?" the model must understand what is meant by a "tyre" and that if there are two tyres on the left hand side of the truck there must be two more on the right hand side.

![tyre image](images/truck.jpg?raw=true)

This points out the weaknesses of the joint embedding approaches that they can only learn the knowledge present in the training set. An alternative to this is to decouple the reasoning _(neural network)_ from the actual storage. Structured representation of knowledge is important for the same. Knowledge bases like DBpedia, Freebase, NELL, ConceptNet etc., store the commonsense and factual information in a machine readable fashion. Each piece of information is stored as a tuple (or a graph) (arg1, rel, arg2) where arg1 and arg1 represent two concepts/ entities/ statements and rel represents the relationship betweeen them.

*__Methods__*: The methods concerning the use of external knowledge bases for visual question answering are presented in the following papers.

## Papers for Methods

### Explicit knowledge based reasoning for visual question answering

### 1. Introduction

This section introduces the reader to the prolems with the existing approaches for visual question answering. There are mainly three kind of problems associated with the VQA systems that do not use external knowledge base for question answering. The typical approach of such kind uses a CNN to find the features from the images and an LSTM to find the textual features and finds a mapping from the question and image pairs to answers.

i) The method cannot explain how it arrived to its answer. This means that it is impossible to tell whether the model is answering based on the vissual information or because of the prevalenace of a particular answer in the training set.

ii) Because the model is trained on individual question answering pairs the range of the questions that can be answered accurately is limited.

iii) The LSTM approach is incapable of explicit reasoning except in very limited situations.

### 2. Dataset Proposal and analysis

This paper proposes a dataset for VQA in external knowledge graphs. Questions to be asked about the image must be reducible to one of the proposed templates.

### 3. VQA approach

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
