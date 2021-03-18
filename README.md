# **Marie**

<p align="center">
  <img width="700" height="448" src="https://github.com/tuhinnn-py/Marie/blob/main/utils/Transformer.jpg">
</p>

### **Overview** 
*Before "**Attention Is All You Need**", Sequence-to-Sequence translations were dependent on complex recurrent architectures like Recurrent Neural Networks(RNNs), Long Short Term Memory(LSTMs) or Gated Recurrent Units(GRUs) with Bidirectional Long Short Term Memory architectures(BiLSTM) being the state-of-the-art model architecture for Natural Language Processing(NLP) tasks.* 

*Infact, some papers even suggested the use of a convolutional architecture to character or word level embeddings to effectively capture grassroot dependencies and relationships depending on the kernel window size(N), thus mimicing a N-gram language model for language modelling tasks. 
However recurrent neural networks come with a lot of problems, some of them being*

- *Recuurent Neural Networks are very difficult to train. Instead of **Stochastic Gradient Descent**, something called a **Truncated Gradient Descent Algorithm** is followed to roughly estimate the gradients for the entire instance, incase of large sentences for example.*
- *Though hypothetically, RNNs are capable of capturing long term **dependencies**(infact in theory, they work fine over an infinite window size as well), RNNs fail to capture long term dependencies. Complex architectures like BiLSTMs and GRUs come as an improvement, but recurrence simply doesn't cover it for large sentences.*
- *Depending on the singular values of weight matrices, gradients seem to explode(**Exploding Gradient Problem**) or diminish to zero(**Vanishing Gradient Problem**).*
- *However the biggest con probably might be the fact that RNNs are not **parallelizable**. Due to their inherent reccurent nature, where the output for the N-1th token serves as an additional input along with the Nth token for the Nth step, RNNs cannot be parallelized.*

*As an improvement to the previously exisiting recurrent architectures, in 2017 Google AI research(Asish Vaswani et. al.) published their groundbreaking transformer architecture in the paper "**Attention Is All You Need**, which is inherently parallelizable and can also capture really long term dependencies due to a mechanism, that the authors call in the paper "**Multi-Head Attention**".*

<p align="center">
  <img width="465" height="565" src="https://github.com/tuhinnn-py/Marie/blob/main/utils/Transformer.png">
</p>

### **Result** 
*Transformers are generally popular and well known for sequence-to-sequence translations. I trained my Transformer implementation on a single Kaggle GPU for 9 hous and the loss converged to 0.07, which according to my opinion is very satisfactory. Within only hours of training, the model mimics a complex English to French translator. With added utilities like a pretrained BERT(which is a deeply bidirectional encoder representation for a transformer), which allows deeply bidirectional contextual token embeddings as input, a satisfactory translator could be roughly be produced within minutes of training.*

<p align="center">
  <img src="https://github.com/tuhinnn-py/Marie/blob/main/utils/chair.png">
</p>

*Here's another generated example.*

<p align="center">
  <img src="https://github.com/tuhinnn-py/Marie/blob/main/utils/hand.png">
</p>

*I wanted to find out the minimum time required to train the transformer model upto a satisfactory benchmark and as it turned out, the minimum number of epochs required were 3 and the estimated training time was only 21 minutes.*

<p align="center">
  <img src="https://github.com/tuhinnn-py/Marie/blob/main/utils/hate.png">
</p>
