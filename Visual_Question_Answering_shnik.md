# Visual Question Answering

[Web](https://visualqa.org)

## Section Wise Explanation

### Section 1: Introduction

This page introduces the reader about the basics of visual question answering and how is it a leading step towards artificial intelligence. Also it tells the author about the two proposed tasks

1. Open ended visual question answering

	- __Input__: An image and a free form natural language question about the image.
	- __Output__: A Open ended answer that might require commonsense other than the information from the image to answer the question.

2. Multiple Choice Task
	
	- __Input__: An image and a free form natural language question about the image and a predefined list of possible answers.
	- __Output__: The most suitable option for the answer of the question

### Section 2: Related Work

This section tells the reader about the related work done befor in the field of visual question answering. This includes:

1. Efforts on Visual Question Answering

	Several papers have explored this field by proposing tasks such as asking some particular kind of questions based on videos and images. An interesting paper is [Don’t Just Listen, Use Your Imagination: Leveraging Visual Common Sense for Non-Visual Tasks](https://arxiv.org/pdf/1502.06108.pdf), which generates abstract scenes to capture visual commonsense	relevant to answering fill in the blanks and paraphrase questions. Other papers have tried to create latent feature vectors using the CNN features extracted from the images and the LSTM features from the sentences. 

2. Text based Question and Answering
	
	Textual question answering tasks provide inspiration for designing the visual question answering tasks. The key concern in text is the grounding of the questions (on what things/ about what are the questions based on?) In VQA naturally the questions are grounded in images.

3. Describing Visual Content and other vision + language tasks

	Several papers have explored the tasks of describing the visual information in terms of referring expressions, coreference resolutions etc., this results in the system only able to capture a few visual concepts. The proposed task on the other hand requires a system to extract more of relavant information from the image. Video and image captioning are also relevant tasks regarding vision and language.


> With the help of this task, it is aimed that the model understands and learns the visual concepts in order to gather the information required to answer the questions with the help of commonsense.


### Section 3: VQA Dataset Collection

![Illustration of the dataset](images/illustration.png?raw=true)

1. Images

	- __Real Images__: Training, validation and test images are taken from the MS COCO dataset, because, this dataset contains the images containing contextual rich information that might be helpful for creating questions for the VQA task.
	- __Abstract Scenes__: To attract the researchers interested in exploring the high-level reasoning VQA tasks and not the low level computer vison tasks, abstract scenes are created using 20 different paperdolls .
	- __Splits__: For real images, splits same as that in MS COCO are used and for abstract scenes: 20K/10K/20K train, val, test splits are used.

2. Text

	- __Captions__: Collected from the MS COCO dataset.
	- __Questions__: Designing questions is a relatively challanging task since questions other than low level vision questions, for example, commonsense questions, are also to be included for any image. The questions were collected through surveys, after conducting various pilot studies to determine a suitable interface for people to ask questions based on an image.
	- __Answers__: People were asked to answer open ended questions. For testing two modalities for answering the questions are offered. Then, different metrics are designed to evaluate the response of a system on a particular question.


![Illustration of the dataset](images/visualize.png?raw=true)

### Section 4: VQA Dataset Analysis

A detailed analysis of what kind of questions and what type of answers is given in the first two parts of this section. This can be summarized in the following images.

![Illustration of the dataset](images/answers.png?raw=true)

1. Commonsense Knowledge

	An analysis of how many questions require commonsense or in general out of domain knowledge for question answering is given in this part.

2. Captions vs Questions

	Do generic image captions provide enough information to answer the questions asked, this is analyzed in this part.


### Section 5: Visual Question Answering: Baselines and Methods

1. Baselines

	- __Random__: Randomly choose an answer from top 1k answers in VQA train/val dataset
	- __Prior "Yes"__: Always select the most popular answer ("Yes")
	- __Per Q-type Prior__: For open ended, pick the most common answer per question type and for multiple choice pick the answer most similar to that of open ended
	- __K Nearest Neighbours__: For open ended choose the most common from first k nearest neighbours and for multiple choice pick the one that is most similar to that for open ended answer

2. Methods

	The proposed method generated a 2 channel vision + language model that culminates with a softmax over K possible outputs.

	- __Image Channel__: This channel provides an embedding for the image. 2 types of embeddings are experimented.
		
		- __I__: The activations from the last hidden layer of VGG are used as 4096 dimensions embeddings.
		- __norm I__: These are l2 normalized activations from the last hidden layer of VGG.

	- __Question Channel__: This channel provides an embedding for the question. 3 types of embeddings are experimented.

		- __Bag of Words Question__: The top 1,000 words in the questions are used to create a bag-of-words representation. Since there is a strong correlation between the words that start a question and the answer (see Fig. 5), we find the top 10 first, second, and third words of the questions and create a 30 dimensional bag-of-words representation. These features are concatenated to get a 1,030-dim embedding for the question.
	
		- __LSTM Q__: An LSTM with one hidden layer is used to obtain 1024-dim embedding for the question. The embedding obtained from the LSTM is a concatenation of last cell state and last hidden state representations (each being 512-dim) from the hidden layer of the LSTM. Each question word is encoded with 300-dim embedding by a fully-connected layer + tanh non-linearity which is then fed to the LSTM. The input vocabulary to the embedding layer consists of all the question words seen in the training dataset 
	
		- __deeper LSTM Q__: An LSTM with two hidden layers is used to obtain 2048-dim embedding for the question. The embedding obtained from the LSTM is a concatenation of last cell state and last hidden state representations (each being 512-dim) from each of the two hidden layers of the LSTM. Hence 2 (hidden layers) x 2 (cell state and hidden state) x 512 (dimensionality of each of the cell states, as well as hidden states) in Fig. 8. This is followed by a fully-connected layer + tanh non-linearity to transform 2048-dim embedding to 1024-dim. The question words are encoded in the same way as in LSTM Q.

![Illustration of the dataset](images/model.png?raw=true)

### Section 6: VQA Challenge and Workshop

See the web link

### Section 7: Conclusion and Discussion

### Appendix
