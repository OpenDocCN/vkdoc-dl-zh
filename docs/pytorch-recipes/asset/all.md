![](images/978-1-4842-4258-2_CoverFigure.jpg)

ISBN 978-1-4842-4257-5e-ISBN 978-1-4842-4258-2 [https://doi.org/10.1007/978-1-4842-4258-2](https://doi.org/10.1007/978-1-4842-4258-2) Library of Congress Control Number: 2018968538 © Pradeepta Mishra 2019 Apress Standard Trademarked names, logos, and images may appear in this book. Rather than use a trademark symbol with every occurrence of a trademarked name, logo, or image we use the names, logos, and images only in an editorial fashion and to the benefit of the trademark owner, with no intention of infringement of the trademark. The use in this publication of trade names, trademarks, service marks, and similar terms, even if they are not identified as such, is not to be taken as an expression of opinion as to whether or not they are subject to proprietary rights. While the advice and information in this book are believed to be true and accurate at the date of publication, neither the authors nor the editors nor the publisher can accept any legal responsibility for any errors or omissions that may be made. The publisher makes no warranty, express or implied, with respect to the material contained herein. Distributed to the book trade worldwide by Springer Science+Business Media New York, 233 Spring Street, 6th Floor, New York, NY 10013\. Phone 1-800-SPRINGER, fax (201) 348-4505, e-mail orders-ny@springer-sbm.com, or visit www.springeronline.com. Apress Media, LLC is a California LLC and the sole member (owner) is Springer Science + Business Media Finance Inc (SSBM Finance Inc). SSBM Finance Inc is a Delaware corporation.

*I would like to dedicate this book to my dear parents, my lovely wife, Prajna, and my daughter, Priyanshi (Aarya). This work would not have been possible without their inspiration, support, and encouragement.*

Introduction

Development of artificial intelligent products and solutions has recently become a norm; hence, the demand for graph theory–based computational frameworks is on the rise. Making the deep learning models work in real-life applications is possible when the modeling framework is dynamic, flexible, and adaptable to other frameworks.

PyTorch is a recent entrant to the league of graph computation tools/programming languages. Addressing the limitations of previous frameworks, PyTorch promises a better user experience in the deployment of deep learning models, and the creation of advanced models using a combination of convolutional neural networks, recurrent neural networks, LSTMs, and deep neural networks.

PyTorch was created by Facebook’s Artificial Intelligence Research division, which seeks to make the model development process simple, straightforward, and dynamic, so that developers do not have to worry about declaring objects before compiling and executing the model. It is based on the Torch framework and is an extension of Python.

This book is intended for data scientists, natural language processing engineers, artificial intelligence solution developers, existing practitioners working on graph computation frameworks, and researchers of graph theory. This book will get you started with understanding tensor basics, computation, performing arithmetic-based operations, matrix algebra, and statistical distribution-based operations using the PyTorch framework.

Chapters [3](#474315_1_En_3_Chapter.xhtml) and [4](#474315_1_En_4_Chapter.xhtml) provide detailed descriptions on neural network basics. Advanced neural networks, such as convolutional neural networks, recurrent neural networks, and LSTMs are explored. Readers will be able to implement these models using PyTorch functions.

Chapters [5](#474315_1_En_5_Chapter.xhtml) and [6](#474315_1_En_6_Chapter.xhtml) discuss fine-tuning the models, hyper parameter tuning, and the refinement of existing PyTorch models in production. Readers learn how to choose the hyper parameters to fine-tune the model.

In Chapter [7](#474315_1_En_7_Chapter.xhtml) , natural language processing is explained. The deep learning models and their applications in natural language processing and artificial intelligence is one of the most demanding skill sets in the industry. Readers will be able to benchmark the execution and performance of PyTorch implementation in deep learning models to execute and process natural language. They will be able to compare PyTorch with other graph computation–based deep learning programming tools.

Acknowledgments

I would like to thank my wife, Prajna, for her continuous inspiration and support, and sacrificing her weekends just to sit alongside me to help me in completing the book; my daughter, Aarya, for being patient all through my writing time; my father, for his eagerness to know how many chapters I had completed.

A big thank you to Nikhil, Celestin, and Divya, for fast-tracking the whole process and helping me and guiding me in the right direction.

I would like to thank my bosses, Ashish and Saty, for always being supportive of my initiatives in the AI and ML journey, and their continuous motivation and inspiration in writing in the AI space.

### About the Author and About the Technical Reviewer

### About the Author

### About the Technical Reviewer

# 1. Introduction to PyTorch, Tensors, and Tensor Operations

PyTorch has been evolving as a larger framework for writing dynamic models. Because of that, it is very popular among data scientists and data engineers deploying large-scale deep learning frameworks. This book provides a structure for the experts in terms of handling activities while working on a practical data science problem. As evident from applications that we use in our day-to-day lives, there are layers of intelligence embedded with the product features. Those features are enabled to provide a better experience and better services to the user.

The world is moving toward artificial intelligence. There are two main components of it: deep learning and machine learning. Without deep learning and machine learning, it is impossible to visualize artificial intelligence.

PyTorch is the most optimized high-performance tensor library for computation of deep learning tasks on GPUs (graphics processing units) and CPUs (central processing units). The main purpose of PyTorch is to enhance the performance of algorithms in large-scale computing environments. PyTorch is a library based on Python and the Torch tool provided by Facebook’s Artificial Intelligence Research group, which performs scientific computing.

NumPy-based operations on a GPU are not efficient enough to process heavy computations. Static deep learning libraries are a bottleneck for bringing flexibility to computations and speed. From a practitioner’s point of view, PyTorch tensors are very similar to the N-dimensional arrays of a NumPy library based on Python. The PyTorch library provides bridge options for moving a NumPy array to a tensor array, and vice versa, in order to make the library flexible across different computing environments.

The use cases where it is most frequently used include natural language processing, image processing, computer vision, social media data analysis, and sensor data processing. Although PyTorch provides a large collection of libraries and modules for computation, three modules are very prominent.

*   *Autograd* . This module provides functionality for automatic differentiation of tensors. A recorder class in the program remembers the operations and retrieves those operations with a trigger called *backward* to compute the gradients. This is immensely helpful in the implementation of neural network models.

*   *Optim* . This module provides optimization techniques that can be used to minimize the error function for a specific model. Currently, PyTorch supports various advanced optimization methods, which includes Adam, stochastic gradient descent (SGD), and more.

*   *NN* . NN stands for *neural network* model. Manually defining the functions, layers, and further computations using complete tensor operations is very difficult to remember and execute. We need functions that automate the layers, activation functions, loss functions, and optimization functions and provides a layer defined by the user so that manual intervention can be reduced. The NN module has a set of built-in functions that automates the manual process of running a tensor operation.

Industries in which artificial intelligence is applied include banking, financial services, insurance, health care, manufacturing, retail, clinical trials, and drug testing. Artificial intelligence involves classifying objects, recognizing the objects to detecting fraud, and so forth. Every learning system requires three things: input data, processing, and an output layer. Figure [1-1](#474315_1_En_1_Chapter.xhtml#Fig1) explains the relationship between these three topics. If the performance of any learning system improves over time by learning from new examples or data, it is called a *machine learning system*. When a machine learning system becomes too difficult to reflect reality, it requires a deep learning system.

In a deep learning system, more than one layer of a learning algorithm is deployed. In machine learning, we think of supervised, unsupervised, semisupervised, and reinforcement learning systems. A supervised machine-learning algorithm is one where the data is labeled with classes or tagged with outcomes. We show the machine the input data with corresponding tags or labels. The machine identifies the relationship with a function. Please note that this function connects the input to the labels or tags.

In unsupervised learning, we show the machine only the input data and ask the machine to group the inputs based on association, similarities or dissimilarities, and so forth.

In semisupervised learning, we show the machine input features and labeled data or tags; however we ask the machine to predict the untagged outcomes or labels.

In reinforcement learning, we introduce a reward and penalty mechanism, where every correct action is rewarded and every incorrect action is penalized.

In all of these examples of machine learning algorithms, we assume that the dataset is small, because getting massive amounts of tagged data is a challenge, and it takes a lot of time for machine learning algorithms to process large-scale matrix computations. Since machine learning algorithms are not scalable for massive datasets, we need deep learning algorithms.

Figure [1-1](#474315_1_En_1_Chapter.xhtml#Fig1) shows the relationships among artificial intelligence, machine learning, and deep learning. Natural language is an important part of artificial intelligence. We need to develop systems that understand natural language and provide responses to the agent. Let’s take an example of machine translation, where a sentence in language 1 (French) can be converted to language 2 (English), and vice versa. To develop such a system, we need a large collection of English-French bilingual sentences. The corpus requirement is very large, as all the language nuances need to be covered by the model.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Fig1_HTML.png](images/474315_1_En_1_Chapter/474315_1_En_1_Fig1_HTML.png)

Figure 1-1

Relationships among ML, DL, and AI

After preprocessing and feature creation, you can observe hundreds of thousands of features that need to be computed to produce output. If we train a machine learning supervised model, it would take months to run and to produce output. To achieve scalability in this task, we need deep learning algorithms, such as a recurrent neural network. This is how the artificial intelligence is connected to deep learning and machine learning.

There are various challenges in deploying deep learning models that require large volumes of labeled data, faster computing machines, and intelligent algorithms. The success of any deep learning system requires good labeled data and better computing machines because the smart algorithms are already available.

The following are various use cases that require deep learning implementation:

*   Speech recognition

*   Video analysis

*   Anomaly detection from videos

*   Natural language processing

*   Machine translation

*   Speech-to-text conversion

The development of the NVIDIA GPU computing for processing large-scale data is another path-breaking innovation. The programming language that is required to run in a GPU environment requires a different programming framework. Two major frameworks are very popular for implementing graphical computing: TensorFlow and PyTorch. In this book, we discuss PyTorch as a framework to implement data science algorithms and make inferences.

The major frameworks for graph computations include PyTorch, TensorFlow, and MXNet. PyTorch and TensorFlow compete with each other in neurocomputations. TensorFlow and PyTorch are equally good in terms of performance; however, the real differences are known only when we benchmark a particular task. Concept-wise there are certain differences.

*   In TensorFlow, we have to define the tensors, initialize the session, and keep placeholders for the tensor objects; however, we do not have to do these operations in PyTorch.

*   In TensorFlow, let’s consider sentiment analysis as an example. Input sentences are tagged with positive or negative tags. If the input sentence’s length is not equal, then we set the maximum sentence length and add zero to make the length of other sentences equal, so that the recurrent neural network can function; however, this is a built-in functionality in PyTorch, so we do not have to define the length of the sentences.

*   In PyTorch, the debugging is much easier and simpler, but it is a difficult task in TensorFlow.

*   In terms of data visualization, model deployment definitely better in TensorFlow; however, PyTorch is evolving and we expect to eventually see the same functionality in the future.

TensorFlow has definitely undergone many changes to reach a stable state. PyTorch is just entering the game, so it will take some time to realize the full potential of this tool.

## What Is PyTorch?

PyTorch is a machine learning and deep learning tool developed by Facebook’s artificial intelligence division to process large-scale image analysis, including object detection, segmentation and classification. It is not limited to these tasks, however. It can be used with other frameworks to implement complex algorithms. It is written using Python and the C++ language. To process large-scale computations in a GPU environment, the programming languages should be modified accordingly. PyTorch provides a great framework to write functions that automatically run in a GPU environment.

## PyTorch Installation

Installing PyTorch is quite simple. In Windows, Linux, or macOS, it is very simple to install if you are familiar with the Anaconda and Conda environments for managing packages. The following steps describe how to install PyTorch in Windows/macOS/Linux environments.

1.  Open the Anaconda navigator and go to the environment page, as displayed in the screenshot shown in Figure [1-2](#474315_1_En_1_Chapter.xhtml#Fig2).

    ![../images/474315_1_En_1_Chapter/474315_1_En_1_Fig2_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Fig2_HTML.jpg)

    Figure 1-2

    Relationships among ML, DL, and AI

2.  Open the terminal and terminal and type the following:

    ```
    conda install -c peterjc123 pytorch
    ```

3.  Launch Jupyter and open the IPython Notebook.

4.  Type the following command to check whether the PyTorch is installed or not.

1.  Check the version of the PyTorch.

```
from __future__ import print_function
import torch
```

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figa_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figa_HTML.jpg)

This installation process was done using a Microsoft Windows machine. The process may vary by operating system, so please use the following URLs for any issue regarding installation and errors.

There are two ways to install it: Conda (Anaconda) library management or the Pip3 package management framework. Also, installations for a local system (such as macOS, Windows, or Linux) and a cloud machine (such as Microsoft Azure, AWS, and GCP) are different. To set up according to your platform, please follow the official PyTorch installation documents at [`https://PyTorch.org/get-started/cloud-partners/`](https://pytorch.org/get-started/cloud-partners/) .

PyTorch has various components.

*   Torch has functionalities similar to NumPy with GPU support.

*   Autograd’s torch.autograd provides classes, methods, and functions for implementing automatic differentiation of arbitrary scalar valued functions. It requires minimal changes to the existing code. You only need to declare `class:'Tensor's`, for which gradients should be computed with the `requires_grad=True` keyword.

*   NN is a neural network library in PyTorch.

*   Optim provides optimization algorithms that are used for the minimization and maximization of functions.

*   Multiprocessing is a useful library for memory sharing between multiple tensors.

*   Utils has utility functions to load data; it also has other functions.

Now we are ready to proceed with the chapter.

## Recipe 1-1\. Using Tensors

### Problem

The data structure used in PyTorch is graph based and tensor based, therefore, it is important to understand basic operations and defining tensors.

### Solution

The solution to this problem is practicing on the tensors and its operations, which includes many examples that use various operations. Though it is assumed that the user is familiar with PyTorch and Python basics, a refresher on PyTorch is essential to create interest among new users.

### How It Works

Let’s have a look at the following examples of tensors and tensor operation basics, including mathematical operations.

The `x` object is a list. We can check whether an object in Python is a tensor object by using the following syntax. Typically, the `is_tensor` function checks and the `is_storage` function checks whether the object is stored as tensor object.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figb_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figb_HTML.jpg)

Now, let’s create an object that contains random numbers from Torch, similar to NumPy library. We can check the tensor and storage type.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figc_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figc_HTML.jpg)

The `y` object is a tensor; however, it is not stored. To check the total number of elements in the input tensor object, the numerical element function can be used. The following script is another example of creating zero values in a 2D tensor and counting the numerical elements in it.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figd_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figd_HTML.jpg)

Like NumPy operations, the eye function creates a diagonal matrix, of which the diagonal elements have ones, and off diagonal elements have zeros. The eye function can be manipulated by providing the shape option. The following example shows how to provide the shape parameter.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Fige_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Fige_HTML.jpg)

Linear space and points between the linear space can be created using tensor operations. Let’s use an example of creating 25 points in a linear space starting from value 2 and ending with 10\. Torch can read from a NumPy array format.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figf_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figf_HTML.jpg)

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figg_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figg_HTML.jpg)

Like linear spacing, logarithmic spacing can be created.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figh_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figh_HTML.jpg)

Random number generation is a common process in data science to generate or gather sample data points in a space to simulate structure in the data. Random numbers can be generated from a statistical distribution, any two values, or a predefined distribution. Like NumPy functions, the random number can be generated using the following example. Uniform distribution is defined as a distribution where each outcome has equal probability of happening; hence, the event probabilities are constant.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figi_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figi_HTML.jpg)

The following script shows how the random number from two values, 0 and 1, are selected. The result tensor can be reshaped to create a (4,5) matrix. The random numbers from a normal distribution with arithmetic mean 0 and standard deviation 1 can also be created, as follows.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figj_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figj_HTML.jpg)

To select random values from a range of values using random permutation requires defining the range first. This range can be created by using the arrange function. When using the arrange function, you must define the step size, which places all the values in an equal distance space. By default, the step size is 1.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figk_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figk_HTML.jpg)

To find the minimum and maximum values in a 1D tensor, argmin and argmax can be used. The dimension needs to be mentioned if the input is a matrix in order to search minimum values along rows or columns.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figl_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figl_HTML.jpg)

If it is either a row or column, it is a single dimension and is called a *1D tensor* . If the input is a matrix, in which rows and columns are present, it is called a *2D tensor* . If it is more than two-dimensional, it is called a *multidimensional tensor* .

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figm_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figm_HTML.jpg)

Now, let’s create a sample 2D tensor and perform indexing and concatenation by using the concat operation on the tensors.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Fign_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Fign_HTML.jpg)

The sample `x` tensor can be used in 3D as well. Again, there are two different options to create three-dimensional tensors; the third dimension can be extended over rows or columns.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figo_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figo_HTML.jpg)

A tensor can be split between multiple chunks. Those small chunks can be created along dim rows and dim columns. The following example shows a sample tensor of size (4,4). The chunk is created using the third argument in the function, as 0 or 1.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figp_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figp_HTML.jpg)

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figq_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figq_HTML.jpg)

The gather function collects elements from a tensor and places it in another tensor using an index argument. The index position is determined by the LongTensor function in PyTorch.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figr_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figr_HTML.jpg)

The LongTensor function or the index select function can be used to fetch relevant values from a tensor. The following sample code shows two options: selection along rows and selection along columns. If the second argument is 0, it is for rows. If it is 1, then it is along the columns.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figs_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figs_HTML.jpg)

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figt_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figt_HTML.jpg)

It is a common practice to check non-missing values in a tensor, the objective is to identify non-zero elements in a large tensor.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figu_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figu_HTML.jpg)

Restructuring the input tensors into smaller tensors not only fastens the calculation process, but also helps in distributed computing. The split function splits a long tensor into smaller tensors.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figv_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figv_HTML.jpg)

Now, let’s have a look at examples of how the input tensor can be resized given the computational difficulty. The transpose function is primarily used to reshape tensors. There are two ways of writing the transpose function: `.t` and `.transpose`.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figw_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figw_HTML.jpg)

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figx_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figx_HTML.jpg)

The unbind function removes a dimension from a tensor. To remove the dimension row, the 0 value needs to be passed. To remove a column, the 1 value needs to be passed.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figy_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figy_HTML.jpg)

Mathematical functions are the backbone of implementing any algorithm in PyTorch; therefore, it is needed to go through functions that help perform arithmetic-based operations. A scalar is a single value, and a tensor 1D is a row, like NumPy. The scalar multiplication and addition with a 1D tensor are done using the add and mul functions.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figz_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figz_HTML.jpg)

The following script shows scalar addition and multiplication with a tensor.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figaa_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figaa_HTML.jpg)

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figab_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figab_HTML.jpg)

Combined mathematical operations, such as expressing linear equations as tensor operations can be done using the following sample script. Here we express the outcome `y` object as a linear combination of beta values times the independent `x` object, plus the constant term.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figac_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figac_HTML.jpg)

Output = Constant + (beta * Independent)

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figad_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figad_HTML.jpg)

Like NumPy operations, the tensor values must be rounded up by using either the ceiling or the flooring function, which is done using the following syntax.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figae_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figae_HTML.jpg)

Limiting the values of any tensor within a certain range can be done using the minimum and maximum argument and using the clamp function. The same function can apply minimum and maximum in parallel or any one of them to any tensor, be it 1D or 2D; 1D is the far simpler version. The following example shows the implementation in a 2D scenario.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figaf_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figaf_HTML.jpg)

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figag_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figag_HTML.jpg)

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figah_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figah_HTML.jpg)

How do we get the exponential of a tensor? How do we get the fractional portion of the tensor if it has decimal places and is defined as a floating data type?

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figai_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figai_HTML.jpg)

The following syntax explains the logarithmic values in a tensor. The values with a negative sign are converted to nan. The power function computes the exponential of any value in a tensor.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figaj_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figaj_HTML.jpg)

To compute the transformation functions (i.e., sigmoid, hyperbolic tangent, radial basis function, and so forth, which are the most commonly used transfer functions in deep learning), you must construct the tensors. The following sample script shows how to create a sigmoid function and apply it on a tensor.

![../images/474315_1_En_1_Chapter/474315_1_En_1_Figak_HTML.jpg](images/474315_1_En_1_Chapter/474315_1_En_1_Figak_HTML.jpg)

## Conclusion

This chapter is a refresher for people who have prior experience in PyTorch and Python. It is a basic building block for people who are new to the PyTorch framework. Before starting the advanced topics, it is important to become familiar with the terminology and basic syntaxes. The next chapter is on using PyTorch to implement probabilistic models, which includes the creation of random variables, the application of statistical distributions, and making statistical inferences.

# 2. Probability Distributions Using PyTorch

Probability and random variables are an integral part of computation in a graph-computing platform like PyTorch. Understanding probability and associated concepts are essential. This chapter covers probability distributions and implementation using PyTorch, and interpreting the results from tests.

In probability and statistics, a random variable is also known as a *stochastic variable* , whose outcome is dependent on a purely stochastic phenomenon, or random phenomenon. There are different types of probability distributions, including normal distribution, binomial distribution, multinomial distribution, and Bernoulli distribution. Each statistical distribution has its own properties.

The torch.distributions module contains probability distributions and sampling functions. Each distribution type has its own importance in a computational graph. The distributions module contains binomial, Bernoulli, beta, categorical, exponential, normal, and Poisson distributions.

## Recipe 2-1\. Sampling Tensors

### Problem

Weight initialization is an important task in training a neural network and any kind of deep learning model, such as a convolutional neural network (CNN), a deep neural network (DNN), and a recurrent neural network (RNN). The question always remains on how to initialize the weights.

### Solution

Weight initialization can be done by using various methods, including random weight initialization. Weight initialization based on a distribution is done using uniform distribution, Bernoulli distribution, multinomial distribution, and normal distribution. How to do it using PyTorch is explained next.

### How It Works

To execute a neural network, a set of initial weights needs to be passed to the backpropagation layer to compute the loss function (and hence, the accuracy can be calculated). The selection of a method depends on the data type, the task, and the optimization required for the model. Here we are going to look at all types of approaches to initialize weights.

If the use case requires reproducing the same set of results to maintain consistency, then a manual seed needs to be set.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figa_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figa_HTML.jpg)

The seed value can be customized. The random number is generated purely by chance. Random numbers can also be generated from a statistical distribution. The probability density function of the *continuous uniform distribution* is defined by the following formula.

![$$ f(x)=\Big\{{\displaystyle \begin{array}{cc}\frac{1}{b-a}&amp; \mathrm{for}\ a\le x\le b,\\ {}0&amp; \mathrm{for}\ x&lt;a\ or\ x&gt;b\end{array}} $$](images/474315_1_En_2_Chapter/474315_1_En_2_Chapter_TeX_Equa.png)

The function of *x* has two points, *a* and *b*, in which *a* is the starting point and *b* is the end. In a continuous uniform distribution, each number has an equal chance of being selected. In the following example, the start is 0 and the end is 1; between those two digits, all 16 elements are selected randomly.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figb_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figb_HTML.jpg)

In statistics, the *Bernoulli distribution* is considered as the discrete probability distribution, which has two possible outcomes. If the event happens, then the value is 1, and if the event does not happen, then the value is 0.

For *discrete probability distribution* , we calculate probability mass function instead of probability density function. The probability mass function looks like the following formula.

![$$ \Big\{{\displaystyle \begin{array}{cc}q=\left(1-p\right)&amp; \mathrm{for}\kern0.125em k=0\\ {}p&amp; \mathrm{for}\kern0.125em k=1\end{array}} $$](images/474315_1_En_2_Chapter/474315_1_En_2_Chapter_TeX_Equb.png)

From the Bernoulli distribution, we create sample tensors by considering the uniform distribution of size 4 and 4 in a matrix format, as follows.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figc_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figc_HTML.jpg)

The generation of sample random values from a *multinomial distribution* is defined by the following script. In a multinomial distribution, we can choose with a replacement or without a replacement. By default, the multinomial function picks up without a replacement and returns the result as an index position for the tensors. If we need to run it with a replacement, then we need to specify that while sampling.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figd_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figd_HTML.jpg)

Sampling from multinomial distribution with a replacement returns the tensors’ index values.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Fige_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Fige_HTML.jpg)

The weight initialization from the normal distribution is a method that is used in fitting a neural network, fitting a deep neural network, and CNN and RNN. Let’s have a look at the process of creating a set of random weights generated from a normal distribution.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figf_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figf_HTML.jpg)

## Recipe 2-2\. Variable Tensors

### Problem

What is a variable in PyTorch and how is it defined? What is a random variable in PyTorch?

### Solution

In PyTorch, the algorithms are represented as a computational graph. A variable is considered as a representation around the tensor object, corresponding gradients, and a reference to the function from where it was created. For simplicity, gradients are considered as slope of the function. The slope of the function can be computed by the derivative of the function with respect to the parameters that are present in the function. For example, in linear regression (Y = W*X + alpha), representation of the variable would look like the one shown in Figure [2-2](#474315_1_En_2_Chapter.xhtml#Fig2).

Basically, a PyTorch variable is a node in a computational graph, which stores data and gradients. When training a neural network model, after each iteration, we need to compute the gradient of the loss function with respect to the parameters of the model, such as weights and biases. After that, we usually update the weights using the gradient descent algorithm. Figure [2-1](#474315_1_En_2_Chapter.xhtml#Fig1) explains how the linear regression equation is deployed under the hood using a neural network model in the PyTorch framework.

In a computational graph structure, the sequencing and ordering of tasks is very important. The one-dimensional tensors are X, Y, W, and alpha in Figure [2-2](#474315_1_En_2_Chapter.xhtml#Fig2). The direction of the arrows change when we implement backpropagation to update the weights to match with Y, so that the error or loss function between Y and predicted Y can be minimized.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Fig1_HTML.png](images/474315_1_En_2_Chapter/474315_1_En_2_Fig1_HTML.png)

Figure 2-1

A sample computational graph of a PyTorch implementation

### How It Works

An example of how a variable is used to create a computational graph is displayed in the following script. There are three variable objects around tensors— x1, x2, and x3—with random points generated from *a* = 12 and *b* = 23\. The graph computation involves only multiplication and addition, and the final result with the gradient is shown.

The partial derivative of the loss function with respect to the weights and biases in a neural network model is achieved in PyTorch using the Autograd module. Variables are specifically designed to hold the changed values while running a backpropagation in a neural network model when the parameters of the model change. The variable type is just a wrapper around the tensor. It has three properties: data, grad, and function.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figg_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figg_HTML.jpg)

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figh_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figh_HTML.jpg)

## Recipe 2-3\. Basic Statistics

### Problem

How do we compute basic statistics, such as mean, median, mode, and so forth, from a Torch tensor?

### Solution

Computation of basic statistics using PyTorch enables the user to apply probability distributions and statistical tests to make inferences from data. Though the Torch functionality is like that of Numpy, Torch functions have GPU acceleration. Let’s have a look at the functions to create basic statistics.

### How It Works

The mean computation is simple to write for a 1D tensor; however, for a 2D tensor, an extra argument needs to be passed as a mean, median, or mode computation, across which the dimension needs to be specified.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figi_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figi_HTML.jpg)

Median, mode, and standard deviation computation can be written in the same way.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figj_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figj_HTML.jpg)

Standard deviation shows the deviation from the measures of central tendency, which indicates the consistency of the data/variable. It shows whether there is enough fluctuation in data or not.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figk_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figk_HTML.jpg)

## Recipe 2-4\. Gradient Computation

### Problem

How do we compute basic gradients from the sample tensors using PyTorch?

### Solution

We are going to consider a sample datase0074, where two variables (x and y) are present. With the initial weight given, can we computationally get the gradients after each iteration? Let’s take a look at the example.

### How It Works

`x_data` and `y_data` both are lists. To compute the gradient of the two data lists requires computation of a loss function, a forward pass, and running the stuff in a loop.

The forward function computes the matrix multiplication of the weight tensor with the input tensor.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figl_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figl_HTML.jpg)

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figm_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figm_HTML.jpg)

![../images/474315_1_En_2_Chapter/474315_1_En_2_Fign_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Fign_HTML.jpg)

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figo_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figo_HTML.jpg)

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figp_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figp_HTML.jpg)

The following program shows how to compute the gradients from a loss function using the variable method on the tensor.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figq_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figq_HTML.jpg)

## Recipe 2-5\. Tensor Operations

### Problem

How do we compute or perform operations based on variables such as matrix multiplication?

### Solution

Tensors are wrapped within the variable, which has three properties: grad, volatile, and gradient.

### How It Works

Let’s create a variable and extract the properties of the variable. This is required to weight update process requires gradient computation. By using the mm module, we can perform matrix multiplication.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figr_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figr_HTML.jpg)

The following program shows the properties of the variable, which is a wrapper around the tensor.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figs_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figs_HTML.jpg)

## Recipe 2-6\. Tensor Operations

### Problem

How do we compute or perform operations based on variables such as matrix-vector computation, and matrix-matrix and vector-vector calculation?

### Solution

One of the necessary conditions for the success of matrix-based operations is that the length of the tensor needs to match or be compatible for the execution of algebraic expressions.

### How It Works

The tensor definition of a scalar is just one number. A 1D tensor is a vector, and a 2D tensor is a matrix. When it extends to an *n* dimensional level, it can be generalized to only tensors. When performing algebraic computations in PyTorch, the dimension of a matrix and a vector or scalar should be compatible.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figt_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figt_HTML.jpg)

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figu_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figu_HTML.jpg)

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figv_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figv_HTML.jpg)

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figw_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figw_HTML.jpg)

Since the mat1 and the mat2 dimensions are different, they are not compatible for matrix addition or multiplication. If the dimension remains the same, we can multiply them. In the following script, the matrix addition throws an error when we multiply similar dimensions—mat1 with mat1\. We get relevant results.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figx_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figx_HTML.jpg)

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figy_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figy_HTML.jpg)

## Recipe 2-7\. Distributions

### Problem

Knowledge of statistical distributions is essential for weight normalization, weight initialization, and computation of gradients in neural network–based operations using PyTorch. How do we know which distributions to use and when to use them?

### Solution

Each statistical distribution follows a pre-established mathematical formula. We are going to use the most commonly used statistical distributions, their arguments in scenarios of problems.

### How It Works

Bernoulli distribution is a special case of *binomial distribution* , in which the number of trials can be more than one; but in a Bernoulli distribution, the number of experiment or trial remains one. It is a discrete probability distribution of a random variable, which takes a value of 1 when there is probability that an event is a success, and takes a value of 0 when there is probability that an event is a failure. A perfect example of this is tossing a coin, where 1 is heads and 0 is tails. Let’s look at the program.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figz_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figz_HTML.jpg)

The *beta distribution* is a family of continuous random variables defined in the range of 0 and 1\. This distribution is typically used for Bayesian inference analysis.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figaa_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figaa_HTML.jpg)

The binomial distribution is applicable when the outcome is twofold and the experiment is repetitive. It belongs to the family of discrete probability distribution, where the probability of success is defined as 1 and the probability of failure is 0\. The binomial distribution is used to model the number of successful events over many trials.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figab_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figab_HTML.jpg)

In probability and statistics, a categorical distribution can be defined as a generalized Bernoulli distribution, which is a discrete probability distribution that explains the possible results of any random variable that may take on one of the possible categories, with the probability of each category exclusively specified in the tensor.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figac_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figac_HTML.jpg)

A *Laplacian distribution* is a continuous probability distribution function that is otherwise known as a *double exponential distribution* . A Laplacian distribution is used in speech recognition systems to understand prior probabilities. It is also useful in Bayesian regression for deciding prior probabilities.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figad_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figad_HTML.jpg)

A *normal distribution* is very useful because of the property of central limit theorem. It is defined by mean and standard deviations. If we know the mean and standard deviation of the distribution, we can estimate the event probabilities.

![../images/474315_1_En_2_Chapter/474315_1_En_2_Fig2_HTML.png](images/474315_1_En_2_Chapter/474315_1_En_2_Fig2_HTML.png)

Figure 2-2

Normal probability distribution

![../images/474315_1_En_2_Chapter/474315_1_En_2_Figae_HTML.jpg](images/474315_1_En_2_Chapter/474315_1_En_2_Figae_HTML.jpg)

## Conclusion

This chapter discussed sampling distribution and generating random numbers from distributions. Neural networks are the primary focus in tensor-based operations. Any sort of machine learning or deep learning model implementation requires gradient computation, updating weight, computing bias, and continuously updating the bias.

This chapter also discussed the statistical distributions supported by PyTorch and the situations where each type of distribution can be applied.

The next chapter discusses deep learning models in detail. Those deep learning models include convolutional neural networks, recurrent neural networks, deep neural networks, and autoencoder models.

# 3. CNN and RNN Using PyTorch

Probability and random variables are an integral part of computation in a graph-computing platform like PyTorch. Understanding probability and the associated concepts are essential. This chapter covers probability distributions and implementation using PyTorch, as well as how to interpret the results of a test. In probability and statistics, a random variable is also known as a *stochastic variable* , whose outcome is dependent on a purely stochastic phenomenon, or random phenomenon. There are different types of probability distribution, including normal distribution, binomial distribution, multinomial distribution, and the Bernoulli distribution. Each statistical distribution has its own properties.

## Recipe 3-1\. Setting Up a Loss Function

### Problem

How do we set up a loss function and optimize it? Choosing the right loss function increases the chances of model convergence.

### Solution

In this recipe, we use another tensor as the update variable, and introduce the tensors to the sample model and compute the error or loss. Then we compute the rate of change in the loss function to measure the choice of loss function in model convergence.

### How It Works

In the following example, t_c and t_u are two tensors. This can be constructed from any NumPy array.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figa_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figa_HTML.jpg)

The sample model is just a linear equation to make the calculation happen and the loss function defined if the mean square error (MSE) shown next. Going forward in this chapter, we will increase the complexity of the model. For now, this is just a simple linear equation computation.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figb_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figb_HTML.jpg)

Let’s now define the model. The w parameter is the weight tensor, which is multiplied with the t_u tensor. The result is added with a constant tensor, b, and the loss function chosen is a custom-built one; it is also available in PyTorch. In the following example, t_u is the tensor used, t_p is the tensor predicted, and t_c is the precomputed tensor, with which the predicted tensor needs to be compared to calculate the loss function.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figc_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figc_HTML.jpg)

The formula w * t_u + b is the linear equation representation of a tensor-based computation.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figd_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figd_HTML.jpg)

The initial loss value is 1763.88, which is too high because of the initial round of weights chosen. The error in the first round of iteration is backpropagated to reduce the errors in the second round, for which the initial set of weights needs to be updated. Therefore, the rate of change in the loss function is essential in updating the weights in the estimation process.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Fige_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Fige_HTML.jpg)

There are two parameters to update the rate of loss function: the learning rate at the current iteration and the learning rate at the previous iteration. If the delta between the two iterations exceeds a certain threshold, then the weight tensor needs to be updated, else model convergence could happen. The preceding script shows the delta and learning rate values. Currently, these are static values that the user has the option to change.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figf_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figf_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figg_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figg_HTML.jpg)

This is how a simple mean square loss function works in a two-dimensional tensor example, with a tensor size of 10,5.

Let’s look at the following example. The MSELoss function is within the neural network module of PyTorch.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figh_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figh_HTML.jpg)

When we look at the gradient calculation that is used for backpropagation, it is shown as MSELoss.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figi_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figi_HTML.jpg)

## Recipe 3-2\. Estimating the Derivative of the Loss Function

### Problem

How do we estimate the derivative of a loss function?

### Solution

Using the following example, we change the loss function to two times the differences between the input and the output tensors, instead of MSELoss function. The following grad_fn, which is defined as a custom function, shows the user how the final output retrieves the derivative of the loss function.

### How It Works

Let’s look at the following example. In the previous recipe, the last line of the script shows the grad_fn as an object embedded in the output object tensor. In this recipe, we explain how this is computed. grad_fn is a derivative of the loss function with respect to the parameters of the model. This is exactly what we do in the following grad_fn.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figj_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figj_HTML.jpg)

The parameters are the input, bias settings, and the learning rate, and the number of epochs for the model training. The estimation of these parameters provides values to the equation.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figk_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figk_HTML.jpg)

This is what the initial result looks like. Epoch is an iteration that produces a loss value from the loss function defined earlier. The params vector is about coefficients and constants that need to be changed to minimize the loss function. The grad function computes the feedback value to the next epoch. This is just an example. The number of epochs chosen is an iterative task depending on the input data, output data, and choice of loss and optimization functions.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figl_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figl_HTML.jpg)

If we reduce the learning rate, we are able to pass relevant values to the gradient, the parameter updates in a better way, and model convergence becomes quicker.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figm_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figm_HTML.jpg)

The initial results look like as the following. The results are at epoch 5 and the loss value is 29.35, which is much lower than 1763.88 at epoch 0, and corresponding to the epoch, the estimated parameters are 0.24 and –.01, at epoch 100\. These parameter values are optimal.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Fign_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Fign_HTML.jpg)

If we reduce the learning rate a bit, then the process of weight updating will be a little slower, which means that the epoch number needs to be increased in order to find a stable state for the model.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figo_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figo_HTML.jpg)

The following are the results that we observe.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figp_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figp_HTML.jpg)

If we increase the number of epochs, then what happens to the loss function and parameter tensor can be viewed in the following script, in which we print the loss value to find the minimum loss corresponding to the epoch. Then we can extract the best parameters from the model.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figq_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figq_HTML.jpg)

The following are the results.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figr_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figr_HTML.jpg)

The following is the final loss value at the final epoch level.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figs_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figs_HTML.jpg)

At epoch 5000, the loss value is 2.92, which is not going down further; hence, at this iteration level, the tensor output displays 5.36 as the final weight and –17.30 as the final bias. These are the final parameters from the model.

To fine-tune this model in estimating parameters, we can redefine the model and the loss function and apply it to the same example.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figt_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figt_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figu_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figu_HTML.jpg)

Set up the parameters. After completing the training process, we should reset the grad function to None.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figv_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figv_HTML.jpg)

## Recipe 3-3\. Fine-Tuning a Model

### Problem

How do we find the gradients of the loss function by applying an optimization function to optimize the loss function?

### Solution

We’ll use the backward() function.

### How It Works

Let’s look at the following example. The backward() function calculates the gradients of a function with respect to its parameters. In this section, we retrain the model with new set of hyperparameters.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figw_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figw_HTML.jpg)

Reset the parameter grid. If we do not o reset the parameters in an existing session, the error values accumulated from any other session become mixed, so it is important to reset the parameter grid.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figx_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figx_HTML.jpg)

After redefining the model and the loss function, let’s retrain the model.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figy_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figy_HTML.jpg)

We have taken 5000 epochs. We train the parameters in a backward propagation method and get the following results. At epoch 0, the loss value is 80.36\. We try to minimize the loss value as we proceed with the next iteration by adjusting the learning rate. At the final epoch, we observe that the loss value is 2.92, which is same result as before but with a different loss function and using backpropagation.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figz_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figz_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figaa_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figaa_HTML.jpg)

The final model parameters are 5.3671 with a bias of –17.3012.

## Recipe 3-4\. Selecting an Optimization Function

### Problem

How do we optimize the gradients with the function in Recipe 3-3?

### Solution

There are certain functions that are embedded in PyTorch, and there are certain optimization functions that the user has to create.

### How It Works

Let’s look at the following example.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figab_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figab_HTML.jpg)

Each optimization method is unique in solving a problem. We will describe it later.

The Adam optimizer is a first-order, gradient-based optimization of stochastic objective functions. It is based on adaptive estimation of lower-order moments. This is computationally efficient enough for deployment on large datasets. To use torch.optim, we have to construct an optimizer object in our code that will hold the current state of the parameters and will update the parameters based on the computed gradients, moments, and learning rate. To construct an optimizer, we have to give it an iterable containing the parameters and ensure that all the parameters are variables to optimize. Then, we can specify optimizer-specific options, such as the learning rate, weight decay, moments, and so forth.

Adadelta is another optimizer that is fast enough to work on large datasets. This method does not require manual fine-tuning of the learning rate; the algorithm takes care of it internally.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figac_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figac_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figad_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figad_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figae_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figae_HTML.jpg)

Now let’s call the model and loss function out once again and apply them along with the optimization function.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figaf_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figaf_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figag_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figag_HTML.jpg)

Let’s look at the gradient in a loss function. Using the optimization library, we can try to find the best value of the loss function.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figah_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figah_HTML.jpg)

The example has two custom functions and a loss function. We have taken two small tensor values. The new thing is that we have taken the optimizer to find the minimum value.

In the following example, we have chosen Adam as the optimizer.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figai_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figai_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figaj_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figaj_HTML.jpg)

In the preceding code, we computed the optimized parameters and computed the predicted tensors using the actual and predicted tensors. We can display a graph that has a line shown as a regression line.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figak_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figak_HTML.jpg)

Let’s visualize the sample data in graphical form using the actual and predicted tensors.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figal_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figal_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figam_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figam_HTML.jpg)

## Recipe 3-5\. Further Optimizing the Function

### Problem

How do we optimize the training set and test it with a validation set using random samples?

### Solution

We’ll go through the process of further optimization.

### How It Works

Let’s look at the following example. Here we set the number of samples, then we take 20% of the data as validation samples using shuffled_indices. We took random samples of all the records. The objective of the train and validation set is to build a model in a training set, make the prediction on the validation set, and check the accuracy of the model.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figan_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figan_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figao_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figao_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figap_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figap_HTML.jpg)

Now let’s run the train and validation process. We first take the training input data and multiply it by the parameter’s next line. We make a prediction and compute the loss function. Using the same model in third line, we make predictions and then we evaluate the loss function for the validation dataset. In the backpropagation process, we calculate the gradient of the loss function for the training set, and using the optimizer, we update the parameters.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figaq_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figaq_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figar_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figar_HTML.jpg)

The following are the last 10 epochs and their results.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figas_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figas_HTML.jpg)

In the previous step, the gradient was set to true. In the following set, we disable gradient calculation by using the torch.no_grad() function . The rest of the syntax remains same. Disabling gradient calculation is useful for drawing inferences, when we are sure that we will not call `Tensor.backward()` . This reduces memory consumption for computations that would otherwise be `requires_grad=True`.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figat_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figat_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figau_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figau_HTML.jpg)

The last rounds of epochs are displayed in other lines of code, as follows.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figav_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figav_HTML.jpg)

The final parameters are 5.44 and –18.012.

## Recipe 3-6\. Implementing a Convolutional Neural Network (CNN)

### Problem

How do we implement a convolutional neural network using PyTorch?

### Solution

There are various built-in datasets available on torchvision. We are considering the MNIST dataset and trying to build a CNN model.

### How It Works

Let’s look at the following example. As a first step, we set up the hyperparameters . The second step is to set up the architecture. The last step is to train the model and make predictions.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figaw_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figaw_HTML.jpg)

In the preceding code, we are importing the necessary libraries for deploying the convolutional neural network model using the digits dataset. The MNIST digits dataset is the most popular dataset in deep learning for computer vision and image processing.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figax_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figax_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figay_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figay_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figaz_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figaz_HTML.jpg)

Let’s load the dataset using the loader functionality .

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figba_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figba_HTML.jpg)

In convolutional neural network architecture, the input image is converted to a feature set as set by color times height and width of the image. Because of the dimensionality of the dataset, we cannot model it to predict the output. The output layer in the preceding graph has classes such as car, truck, van, and bicycle. The input bicycle image has features that the CNN model should make use of and predict it correctly. The convolution layer is always accompanied by the pooling layer, which can be max pooling and average pooling. The different layers of pooling and convolution continue until the dimensionality is reduced to a level where we can use fully connected simple neural networks to predict the correct classes.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbb_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbb_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbc_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbc_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbd_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbd_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbe_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbe_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbf_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbf_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbg_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbg_HTML.jpg)

In the preceding graph, if we look at the number 4, it is scattered throughout the graph. Ideally, all of the 4s are closer to each other. This is because the test accuracy was very low.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbh_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbh_HTML.jpg)

In this iteration, the training loss is reduced from 0.4369 to 0.1482 and the test accuracy improves from 16% to 94%. The digits with the same color are placed closely on the graph.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbi_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbi_HTML.jpg)

In the next epoch, the test accuracy on the MNIST digits dataset the accuracy increases to 95%.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbj_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbj_HTML.jpg)

In the final step/epoch, the digits with similar numbers are placed together. After training a model successfully, the next step is to make use of the model to predict. The following code explains the predictions process. The output object is numbered as 0, 1, 2, and so forth. The following shows the real and predicted numbers.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbk_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbk_HTML.jpg)

## Recipe 3-7\. Reloading a Model

### Problem

How do we store and re-upload a model that has already been trained? Given the nature of deep learning models, which typically require a larger training time, the computational process creates a huge cost to the company. Can we retrain the model with new inputs and store the model?

### Solution

In the production environment, we typically cannot train and predict at the same time because the training process takes a very long time. The prediction services cannot be applied until the training process using epoch is completed, the prediction services cannot be applied. Disassociating the training process from the prediction process is required; therefore, we need to store the application’s trained model and continue until the next phase of training is done.

### How It Works

Let’s look at the following example, where we are creating the save function, which uses the Torch neural network module to create the model and the restore_net() function to get back the neural network model that was trained earlier.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbl_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbl_HTML.jpg)

The preceding script contains a dependent Y variable and an independent X variable as sample data points to create a neural network model. The following `save` function stores the model. The `net1` object is the trained neural network model, which can be stored using two different protocols: (1) save the entire neural network model with all the weights and biases, and (2) save the model using only the weights. If the trained model object is very heavy in terms of size, we should save only the parameters that are weights; if the trained object size is low, then the entire model can be stored.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbm_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbm_HTML.jpg)

The prebuilt neural network model can be reloaded to the existing PyTorch session by using the `load` function. To test the `net1` object and make predictions, we load the `net1` object and store the model as `net2`. By using the `net2` object, we can predict the outcome variable. The following script generates the graph as a dependent and an independent variable. `prediction.data.numpy()` in the last line of the code shows the predicted result.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbn_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbn_HTML.jpg)

Loading the pickle file format of the entire neural network is relatively slow; however, if we are only making predictions for a new dataset, we can only load the parameters of the model in a pickle format rather than the whole network.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbo_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbo_HTML.jpg)

Reuse the model. The `restore` function makes sure that the trained parameters can be reused by the model. To restore the model, we can use the load_state_dict() function to load the parameters of the model. If we see the following three models in the graph, they are identical, because net2 and net3 are copies of net1.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbp_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbp_HTML.jpg)

## Recipe 3-8\. Implementing a Recurrent Neural Network (RNN)

### Problem

How do we set up a recurrent neural network using the MNIST dataset?

### Solution

The recurrent neural network is considered as a memory network. We will use the epoch as 1 and a batch size of 64 samples at a time to establish the connection between the input and the output. Using the RNN model, we can predict the digits present in the images.

### How It Works

Let’s look at the following example. The recurrent neural network takes a sequence of vectors in the input layer and produces a sequence of vectors in the output layer. The information sequence is processed through the internal state transfer in the recurrent layer. Sometimes the output values have a long dependency in past historical values. This is another variant of the RNN model: the long short-term memory (LSTM) model . This is applicable for any sort of domain where the information is consumed in a sequential manner; for example, in a time series where the current stock price is decided by the historical stock price, where the dependency can be short or long. Similarly, the context prediction using the long and short range of textual input vectors. There are other industry use cases, such as noise classification, where noise is also a sequence of information.

The following piece of code explains the execution of RNN model using PyTorch module.

There are three sets of weights: U, V and W. The set of weights vector, represented by W, is for passing information among the memory cells in the network that display communication among the hidden state. RNN uses an embedding layer using the Word2vec representation. The embedding matrix is the size of the number of words by the number of neurons in the hidden layer. If you have 20,000 words and 1000 hidden units, for example, the matrix has a 20,000×1000 size of the embedding layer. The new representations are passed to LSTM cells, which go to a sigmoid output layer.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbq_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbq_HTML.jpg)

The RNN models have hyperparameters, such as the number of iterations (`EPOCH`); batch size dependent on the memory available in a single machine; a time step to remember the sequence of information; input size, which shows the vector size; and learning rate. The selection of these values is indicative; we cannot depend on them for other use cases. The value selection for hyperparameter tuning is an iterative process; either you can choose multiple parameters and decide which one is working, or do parallel training of the model and decide which one is working fine.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbr_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbr_HTML.jpg)

Using the dsets.MINIST() function, we can load the dataset to the current session. If you need to store the dataset, then download it locally.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbs_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbs_HTML.jpg)

The preceding script shows what the sample image dataset would look like. To train the deep learning model, we need to convert the whole training dataset into mini batches, which help us with averaging the final accuracy of the model. By using the data loader function, we can load the training data and prepare the mini batches. The purpose of the shuffle selection in mini batches is to ensure that the model captures all the variations in the actual dataset.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbt_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbt_HTML.jpg)

The preceding script prepares the training dataset. The test data is captured with the flag `train=False`. It is transformed to a tensor using the test data random sample of 2000 each at a time is picked up for testing the model. The test features set is converted to a variable format and the test label vector is represented in a NumPy array format.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbu_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbu_HTML.jpg)

In the preceding RNN class, we are training an LSTM network , which is proven effective for holding memory for a long time, and thus helps in learning. If we use the nn.RNN() model, it hardly learns the parameters, because the vanilla implementation of RNN cannot hold or remember the information for a long period of time. In the LSTM network, the image width is considered the input size, hidden size is decided as the number of neurons in the hidden layer, `num_layers` shows the number of RNN layers in the network.

The RNN module, within the LSTM module, produces the output as a vector size of 64×10 because the output layer has digits to be classified as 0 to 9\. The last forward function shows how to proceed with forward propagation in an RNN network.

The following script shows how the LSTM model is processed under the RNN class. In the LSTM function, we pass the input length as 28 and the number of neurons in the hidden layer as 64, and from the hidden 64 neurons to the output 10 neurons.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbv_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbv_HTML.jpg)

To optimize all RNN parameters, we use the Adam optimizer. Inside the function, we use the learning rate as well. The loss function used in this example is the cross-entropy loss function. We need to provide multiple epochs to get the best parameters.

In the following script, we are printing the training loss and the test accuracy. After one epoch, the test accuracy increases to 95% and the training loss reduces to 0.24.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbw_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbw_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbx_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbx_HTML.jpg)

Once the model is trained, then the next step is to make predictions using the RNN model. Then we compare the actual vs. real output to assess how the model is performing.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figby_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figby_HTML.jpg)

## Recipe 3-9\. Implementing a RNN for Regression Problems

### Problem

How do we set up a recurrent neural network for regression-based problems?

### Solution

The regression model requires a target function and a feature set, and then a function to establish the relationship between the input and the output. In this example, we are going to use the recurrent neural network (RNN) for a regression task. Regression problems seem to be very simple; they do work best but are limited to data that shows clear linear relationships. They are quite complex when predicting nonlinear relationships between the input and the output.

### How It Works

Let’s look at the following example that shows a nonlinear cyclical pattern between input and output data. In the previous recipe, we looked at an example of RNN in general for classification-related problems, where predicted the class of the input image. In regression, however, the architecture of RNN would change, because the objective is to predict the real valued output. The output layer would have one neuron in regression-related problems.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figbz_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figbz_HTML.jpg)

RNN time step implies that the last 10 values predict the current value, and the rolling happens after that.

The following script shows some sample series in which the target cos function is approximated by the sin function.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figca_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figca_HTML.jpg)

## Recipe 3-10\. Using PyTorch Built-in Functions

### Problem

How do we set up an RNN module and call the RNN function using PyTorch?

### Solution

By using the built-in function available in the neural network module, we can implement an RNN model.

### How It Works

Let’s look at the following example. The neural network module in the PyTorch library contains the RNN function. In the following script, we use the input matrix size, the number of neurons in the hidden layer, and the number of hidden layers in the network.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcb_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcb_HTML.jpg)

After creating the RNN class function, we need to provide the optimization function, which is Adam, and this time, the loss function is the mean square loss function. Since the objective is to make predictions of a continuous variable, we use MSELoss function in the optimization layer.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcc_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcc_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcd_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcd_HTML.jpg)

Now we iterate over 60 steps to predict the cos function generated from the sample space , and have it predicted by a sin function. The iterations take the learning rate defined as before, and backpropagate the error to reduce the MSE and improve the prediction.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figce_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figce_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcf_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcf_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcg_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcg_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figch_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figch_HTML.jpg)

## Recipe 3-11\. Working with Autoencoders

### Problem

How do we perform clustering using the autoencoders function?

### Solution

Unsupervised learning is a branch of machine learning that does not have a target column or the output is not defined. We only need to understand the unique patterns existing in the data. Let’s look at the autoencoder architecture in Figure [3-1](#474315_1_En_3_Chapter.xhtml#Fig1). The input feature space is transformed into a lower dimensional tensor representation using a hidden layer and mapped back to the same input space. The layer that is precisely in the middle holds the autoencoder’s values.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Fig1_HTML.png](images/474315_1_En_3_Chapter/474315_1_En_3_Fig1_HTML.png)

Figure 3-1

Autoencoder architecture

### How It Works

Let’s look at the following example. The torchvision library contains popular datasets, model architectures, and frameworks. Autoencoder is a process of identifying latent features from the dataset; it is used for classification, prediction, and clustering. If we put the input data in the input layer and the same dataset in the output layer, then we add multiple layers of hidden layers with many neurons, and then we pass through a series of epochs. We get a set of latent features in the innermost hidden layer. The weights or parameters in the central hidden layer are known as the *autoencoder* *layer* .

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figci_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figci_HTML.jpg)

We again use the MNIST dataset to experiment with autoencoder functionality. This time we are taking 10 epochs, a batch size 64 to be passed to the network, a learning rate of 0.005, and 5 images for testing.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcj_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcj_HTML.jpg)

The following plot shows the dataset uploaded from the torchvision library and displayed as an image.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figck_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figck_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcl_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcl_HTML.jpg)

Let’s discuss the autoencoder architecture. The input has 784 features. It has a height of 28 and a width of 28\. We pass the 784 neurons from the input layer to the first hidden layer, which has 128 neurons in it. Then we apply the hyperbolic tangent function to pass the information to the next hidden layer. The second hidden layer contains 128 input neurons and transforms it into 64 neurons. In the third hidden layer, we apply the hyperbolic tangent function to pass the information to the next hidden layer. The innermost layer contains three neurons, which are considered as three features, which is the end of the encoder layer. Then the decoder function expands the layer back to the 784 features in the output layer.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcm_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcm_HTML.jpg)

Once we set the architecture, then the normal process of making the loss function minimize corresponding to a learning rate and optimization function happens. The entire architecture passes through a series of epochs in order to reach the target output.

## Recipe 3-12\. Fine-Tuning Results Using Autoencoder

### Problem

How do we set up iterations to fine-tune the results?

### Solution

Conceptually, an autoencoder works the same as the clustering model. In unsupervised learning, the machine learns patterns from data and generalizes it to the new dataset. The learning happens by taking a set of input features. Autoencoder functions are also used for feature engineering.

### How It Works

Let’s look at the following example. The same MNIST dataset is used as an example, and the objective is to understand the role of the epoch in achieving a better autoencoder layer. We increase the epoch size to reduce errors to a minimum; however, in practice, increasing the epoch has many challenges, including memory constraints.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcn_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcn_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figco_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figco_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcp_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcp_HTML.jpg)

By using the encoder function, we can represent the input features into a set of latent features. By using the decoder function, however, we can reconstruct the image. Then we can match how image reconstruction is done by using the autoencoder functions. From the preceding set of graphs, it is clear that as we increase the epoch, the image recognition becomes transparent.

## Recipe 3-13\. Visualizing the Encoded Data in a 3D Plot

### Problem

How do we visualize the MNIST data in a 3D plot?

### Solution

We use the autoencoder function to get the encoded features and then use the dataset to represent it in a 3D plane.

### How It Works

Let’s look at the following example. This recipe is about how to represent the autoencoder function derived from the preceding recipe in the three-dimensional space, because we have three neurons in the innermost hidden layer. The following display shows a three-dimensional neuron.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcq_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcq_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcr_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcr_HTML.jpg)

## Recipe 3-14\. Restricting Model Overfitting

### Problem

When we fit many neurons and layers to predict the target class or output variable, the function usually overfits the training dataset. Because of model overfitting, we cannot make a good prediction on the test set. The test accuracy is not the same as training accuracy. There would be deviations in training and test accuracy.

### Solution

To restrict model overfitting, we consciously introduce dropout rate, which means randomly delete (let’s say) 10% or 20% of the weights in the network, and check the model accuracy at the same time. If we are able to match the same model accuracy after deleting the 10% or 20% of the weights, then our model is good.

### How It Works

Let’s look at the following example. Model overfitting is occurs when the trained model does not generalize to other test case scenarios. It is identified when the training accuracy becomes significantly different from the test accuracy. To avoid model overfitting, we can introduce the dropout rate in the model.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcs_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcs_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figct_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figct_HTML.jpg)

The dropout rate introduction to the hidden layer ensures that weights less than the threshold defined are removed from the architecture. A typical threshold for an application’s dropout rate is 20% to 50%. A 20% dropout rate implies a smaller degree of penalization; however, the 50% threshold implies heavy penalization of the model weights.

In the following script, we apply a 50% dropout rate to drop the weights from the model. We applied the dropout rate twice.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcu_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcu_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcv_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcv_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcw_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcw_HTML.jpg)

The selection of right dropout rate requires a fair idea about the business and domain.

## Recipe 3-15\. Visualizing the Model Overfit

### Problem

Assess model overfitting.

### Solution

We change the model hyperparameters and iteratively see if the model is overfitting data or not.

### How It Works

Let’s look at the following example. The previous recipe covered two types of neural networks: overfitting and dropout rate. When the model parameters estimated from the data come closer to the actual data, for the training dataset and the same models differs from the test set, it is a clear sign of model overfit. To restrict model overfit, we can introduce the dropout rate, which deletes a certain percentage of connections (as in weights from the network) to allow the trained model to come to the real data.

In the following script, the iterations were taken 500 times. The predicted values are generated from the base model, which shows overfitting, and from the dropout model, which shows the deletion of some weights. In the same fashion, we create the two loss functions, backpropagation, and implementation of the optimizer.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcx_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcx_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcy_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcy_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figcz_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figcz_HTML.jpg)

The initial round of plotting includes the overfitting loss and dropout loss and how it is different from the actual training and test data points from the preceding graph.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figda_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figda_HTML.jpg)

After many iterations, the preceding graph was generated by using the two functions with the actual model and with the dropout rate. The takeaway from this graph is that actual training data may get closer to the overfit model; however, the dropout model fits the data really well.

## Recipe 3-16\. Initializing Weights in the Dropout Rate

### Problem

How do we delete the weights in a network? Should we delete randomly or by using any distribution?

### Solution

We should delete the weights in the dropout layer based on probability distribution, rather than randomly.

### How It Works

Let’s look at the following example. In the previous recipe, three layers of a dropout rate were introduced: one after the first hidden layer and two after the second hidden layer. The probability percentage was 0.50, which meant randomly delete 50% of the weights. Sometimes, random selection of weights from the network deletes relevant weights, so an alternative idea is to delete the weights in the network generated from statistical distribution.

The following script shows how to generate the weights from a uniform distribution, then we can use the set of weights in the network architecture.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figdb_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figdb_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figdc_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figdc_HTML.jpg)

## Recipe 3-17\. Adding Math Operations

### Problem

How do we set up the broadcasting function and optimize the convolution function?

### Solution

The script snippet shows how to introduce batch normalization when setting up a convolutional neural network model, and then further setting up a pooling layer.

### How It Works

Let’s look at the following example. To introduce batch normalization in the convolutional layer of the neural network model, we need to perform tensor-based mathematical operations that are functionally different from other methods of computation.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figdd_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figdd_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figde_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figde_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figdf_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figdf_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figdg_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figdg_HTML.jpg)

The following piece of script shows how the batch normalization using a 2D layer is resolved before entering into the 2D max pooling layer.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figdh_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figdh_HTML.jpg)

## Recipe 3-18\. Embedding Layers in RNN

### Problem

The recurrent neural network is used mostly for text processing. An embedded feature offers more accuracy on a standard RNN model than raw features. How do we create embedded features in an RNN?

### Solution

The first step is to create an embedding layer, which is a fixed dictionary and fixed-size lookup table, and then introduce the dropout rate after than create gated recurrent unit.

### How It Works

Let’s look at the following example. When textual data comes in as a sequence, the information is processed in a sequential way; for example, when we describe something, we use a set of words in sequence to convey the meaning. If we use the individual words as vectors to represent the data, the resulting dataset would be very sparse. But if we use a phrase-based approach or a combination of words to represent as feature vector, then the vectors become a dense layer. Dense vector layers are called *word embeddings* , as the embedding layer conveys a context or meaning as the result. It is definitely better than the bag-of-words approach.

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figdi_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figdi_HTML.jpg)

![../images/474315_1_En_3_Chapter/474315_1_En_3_Figdj_HTML.jpg](images/474315_1_En_3_Chapter/474315_1_En_3_Figdj_HTML.jpg)

## Conclusion

This chapter covered using the PyTorch API, creating a simple neural network mode, and optimizing the parameters by changing the hyperparameters (i.e., learning rate, epochs, gradients drop). We looked at recipes on how to create a convolutional neural network and a recurrent neural network, and introduced the dropout rate in these networks to control model overfitting.

We took small tensors to follow what exactly goes on behind the scenes with calculations and so forth. We only need to define the problem statement, create features, and apply the recipe to get results. In the next chapter, we implement many more examples with PyTorch.

# 4. Introduction to Neural Networks Using PyTorch

Deep neural network–based models are gradually becoming the backbone for artificial intelligence and machine learning implementations. The future of data mining will be governed by the usage of artificial neural network–based advanced modeling techniques. One obvious question is why neural networks are only now gaining so much importance, because it was invented in 1950s.

Borrowed from the computer science domain, neural networks can be defined as a parallel information processing system where all the input relates to each other, like neurons in the human brain, to transmit information so that activities like face recognition, image recognition, and so forth, can be performed. In this chapter, you learn about the application of neural network-based methods on various data mining tasks, such as classification, regression, forecasting, and feature reduction. An *artificial neural network* (ANN) functions in a way that is similar to the way that the human brain functions, in which billions of neurons link to each other for information processing and insight generation.

## Recipe 4-1\. Working with Activation Functions

### Problem

What are the activation functions and how do they work in real projects? How do you implement an activation function using PyTorch?

### Solution

Activation function is a mathematical formula that transforms a vector available in a binary, float, or integer format to another format based on the type of mathematical transformation function. The neurons are present in different layers—input, hidden, and output, which are interconnected through a mathematical function called an *activation function*. There are different variants of activation functions, which are explained next. Understanding the activation function helps in accurately implementing a neural network model.

### How It Works

All the activation functions that are part of a neural network model can be broadly classified as linear functions and nonlinear functions. The PyTorch torch.nn module creates any type of a neural network model. Let’s look at some examples of the deployment of activation functions using PyTorch and the torch.nn module.

The core differences between PyTorch and TensorFlow is the way a computational graph is defined, the way the two frameworks perform calculations, and the amount of flexibility we have in changing the script and introducing other Python-based libraries in it. In TensorFlow, we need to define the variables and placeholders before we initialize the model. We also need to keep track of objects that we need later, and for that we need a placeholder. In TensorFlow, we need to define the model first, and then compile and run; however, in PyTorch, we can define the model as we go—we don’t have to keep placeholders in the code. That’s why the PyTorch framework is dynamic.

#### Linear Function

A linear function is a simple functions typically used to transfer information from the demapping layer to the output layer. We use the linear function in places where variations in data are lower. In a deep learning model, practitioners typically use a linear function in the last hidden layer to the output layer. In the linear function, the output is always confined to a specific range; because of that, it is used in the last hidden layer in a deep learning model, or in linear regression–based tasks, or in a deep learning model where the task is to predict the outcome from the input dataset. The following is the formula.

![$$ y=\alpha +\beta x $$](images/474315_1_En_4_Chapter/474315_1_En_4_Chapter_TeX_Equa.png)

#### Bilinear Function

A bilinear function is a simple functions typically used to transfer information. It applies a bilinear transformation to incoming data.

![$$ y={x}_1\ast A\ast {x}_2+b $$](images/474315_1_En_4_Chapter/474315_1_En_4_Chapter_TeX_Equb.png)

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figa_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figa_HTML.jpg)

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figb_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figb_HTML.jpg)

#### Sigmoid Function

A sigmoid function is frequently used by professionals in data mining and analytics because it is easier to explain and implement. It is a nonlinear function. When we pass weights from the input layer to the hidden layer in a neural network, we want our model to capture all sorts of nonlinearity present in the data; hence, using the sigmoid function in the hidden layers of a neural network is recommended. The nonlinear functions help with generalizing the dataset. It is easier to compute the gradient of a function using a nonlinear function.

The sigmoid function is a specific nonlinear activation function. The sigmoid function output is always confined within 0 and 1; therefore, it is mostly used in performing classification-based tasks. One of the limitations of the sigmoid function is that it may get stuck in local minima. An advantage is that it provides probability of belonging to the class. The following is its equation.

![$$ f(x)=\frac{1}{1+{e}^{-\beta x}} $$](images/474315_1_En_4_Chapter/474315_1_En_4_Chapter_TeX_Equc.png)

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figc_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figc_HTML.jpg)

#### Hyperbolic Tangent Function

A hyperbolic tangent function is another variant of a transformation function. It is used to transform information from the mapping layer to the hidden layer. It is typically used between the hidden layers of a neural network model. The range of the tanh function is between –1 and +1.

![$$ \tanh (x)=\frac{e^x-{e}^{-x}}{e^x+{e}^{-x}} $$](images/474315_1_En_4_Chapter/474315_1_En_4_Chapter_TeX_Equd.png)

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figd_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figd_HTML.jpg)

![../images/474315_1_En_4_Chapter/474315_1_En_4_Fige_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Fige_HTML.jpg)

#### Log Sigmoid Transfer Function

The following formula explains the log sigmoid transfer function, which is used in mapping the input layer to the hidden layer. If the data is not binary, and it is a float type with a lot of outliers (as in large numeric values present in the input feature), then we should use the log sigmoid transfer function.

![$$ f(x)=\log \left(\frac{1}{1+{e}^{-\beta x}}\right) $$](images/474315_1_En_4_Chapter/474315_1_En_4_Chapter_TeX_IEq1.png)

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figf_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figf_HTML.jpg)

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figg_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figg_HTML.jpg)

#### ReLU Function

The rectified linear unit (ReLu) is another activation function. It is used in transferring information from the input layer to the output layer. ReLu is mostly used in a convolutional neural network model. The range in which this activation function operates is from 0 to infinity. It is mostly used between different hidden layers in a neural network model.

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figh_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figh_HTML.jpg)

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figi_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figi_HTML.jpg)

The different types of transfer functions are interchangeable in a neural network architecture. They can be used in different stages , such as the input to the hidden layer, the hidden layer to the output layer, and so forth, to improve the model’s accuracy.

#### Leaky ReLU

In a standard neural network model, a dying gradient problem is common. To avoid this issue, leaky ReLU is applied. Leaky ReLU allows a small and non-zero gradient when the unit is not active.

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figj_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figj_HTML.jpg)

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figk_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figk_HTML.jpg)

## Recipe 4-2\. Visualizing the Shape of Activation Functions

### Problem

How do we visualize the activation functions? The visualization of activation functions is important in correctly building a neural network model.

### Solution

The activation functions translate the data from one layer into another layer. The transformed data can be plotted against the actual tensor to visualize the function. We have taken a sample tensor, converted it to a PyTorch variable, applied the function, and stored it as another tensor. Represent the actual tensor and the transformed tensor using matplotlib.

### How It Works

The right choice of an activation function will not only provide better accuracy but also help with extracting meaningful information.

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figl_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figl_HTML.jpg)

In this script, we have an array in the linear space between –10 and +10, and we have 1500 sample points. We converted the vector to a Torch variable, and then made a copy as a NumPy variable for plotting the graph. Then, we calculated the activation functions. The following images show the activation functions.

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figm_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figm_HTML.jpg)

![../images/474315_1_En_4_Chapter/474315_1_En_4_Fign_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Fign_HTML.jpg)

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figo_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figo_HTML.jpg)

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figp_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figp_HTML.jpg)

## Recipe 4-3\. Basic Neural Network Model

### Problem

How do we build a basic neural network model using PyTorch?

### Solution

A basic neural network model in PyTorch requires six steps: preparing training data, initializing weights, creating a basic network model, calculating the loss function, selecting the learning rate, and optimizing the loss function with respect to the model’s parameters.

### How It Works

Let’s follow a step-by-step approach to create a basic neural network model.

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figq_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figq_HTML.jpg)

To show a sample neural network model, we prepare the dataset and change the data type to a float tensor. When we work on a project, data preparation for building it is a separate activity. Data preparation should be done in the proper way. In the preceding step, train x and train y are two NumPy vectors. Next, we change the data type to a float tensor because it is necessary for matrix multiplication. The next step is to convert it to variable, because a variable has three properties that help us fine-tune the object. In the dataset, we have 17 data points on one dimension.

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figr_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figr_HTML.jpg)

The set_weight() function initializes the random weights that the neural network model will use in forward propagation. We need two tensors weights and biases. The build_network() function simply multiplies the weights with input, adds the bias to it, and generates the predicted values. This is a custom function that we built. If we need to implement the same thing in PyTorch, then it is much simpler to use nn.Linear() when we need to use it for linear regression.

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figs_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figs_HTML.jpg)

Once we define a network structure, then we need to compare the results with the output to assess the prediction step. The metric that tracks the accuracy of the system is the loss function, which we want to be minimal. The loss function may have a different shape. How do we know exactly where the loss is at a minimum, which corresponds to which iteration is providing the best results? To know this, we need to apply the optimization function on the loss function; it finds the minimum loss value. Then we can extract the parameters corresponding to that iteration.

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figt_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figt_HTML.jpg)

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figu_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figu_HTML.jpg)

Median, mode and standard deviation computation can be written in the sa

Standard deviation shows the deviation from the measures of central tendency, which indicates the consistency of the data/variable. It shows whether there is enough fluctuation in data or not.

## Recipe 4-4\. Tensor Differentiation

### Problem

What is tensor differentiation, and how is it relevant in computational graph execution using the PyTorch framework?

### Solution

The computational graph network is represented by nodes and connected through functions. There are two different kinds of nodes: dependent and independent. *Dependent nodes* are waiting for results from other nodes to process the input. *Independent nodes* are connected and are either constants or the results. Tensor differentiation is an efficient method to perform computation in a computational graph environment.

### How It Works

In a computational graph, tensor differentiation is very effective because the tensors can be computed as parallel nodes, multiprocess nodes, or multithreading nodes. The major deep learning and neural computation frameworks include this tensor differentiation.

Autograd is the function that helps perform tensor differentiation, which means calculating the gradients or slope of the error function, and backpropagating errors through the neural network to fine-tune the weights and biases. Through the learning rate and iteration, it tries to reduce the error value or loss function.

To apply tensor differentiation, the nn.backward() method needs to be applied. Let’s take an example and see how the error gradients are backpropagated. To update the curve of the loss function, or to find where the shape of the loss function is minimum and in which direction it is moving, a derivative calculation is required. Tensor differentiation is a way to compute the slope of the function in a computational graph.

![../images/474315_1_En_4_Chapter/474315_1_En_4_Figv_HTML.jpg](images/474315_1_En_4_Chapter/474315_1_En_4_Figv_HTML.jpg)

In this script, the x is a sample tensor , for which automatic gradient calculation needs to happen. The fn is a linear function that is created using the x variable. Using the backward function, we can perform a backpropagation calculation. The .grad() function holds the final output from the tensor differentiation.

## Conclusion

This chapter discussed various activation functions and the use of the activation functions in various situations. The method or system to select the best activation function is accuracy driven; the activation function that gives the best results should always be used dynamically in the model. We also created a basic neural network model using small sample tensors, updated the weights using optimization, and generated predictions. In the next chapter, we see more examples.

# 5. Supervised Learning Using PyTorch

Supervised machine learning is the most sophisticated branch of machine learning. It is in use in almost all fields, including artificial intelligence, cognitive computing, and language processing. Machine learning literature broadly talks about three types of learning: supervised, unsupervised, and reinforcement learning. In supervised learning, the machine learns to recognize the output; hence, it is task driven and the task can be classification or regression.

In unsupervised learning, the machine learns patterns from data; thus, it generalizes the new dataset and the learning happens by taking a set of input features. In reinforcement learning, the learning happens in response to a system that reacts to situations.

This chapter covers regression techniques in detail with a machine learning approach and interprets the output from regression methods in the context of a business scenario. The algorithmic classification is shown in Figure [5-1](#474315_1_En_5_Chapter.xhtml#Fig1).

![../images/474315_1_En_5_Chapter/474315_1_En_5_Fig1_HTML.png](images/474315_1_En_5_Chapter/474315_1_En_5_Fig1_HTML.png)

Figure 5-1

Algorithmic classification

Each object or row represents one event and each event is categorized into groups. Identifying which level group a record belongs to is called *classification*, in which the target variable has specific labels or tags attached to the events. For example, in a bank database, each customer is tagged as either a loyal customer or not a loyal customer. In a medical records database, each patient’s disease is tagged. In the telecom industry, each subscriber is tagged as a churn or non-churn customer. These are examples in which a supervised algorithm performs classification. The word *classification* comes from the classes available in the target column.

In *regression learning* , the objective is to predict the value of a continuous variable; for example, given the features of a property, such as the number of bedrooms, square feet, nearby areas, the township, and so forth, the asking price for the house is determined. In such scenarios, the regression models can be used. Similar examples include predicting stock prices or the sales, revenue, and profit of a business.

In an unsupervised learning algorithm, we do not have an outcome variable, and tagging or labeling is not available. We are interested in knowing the natural grouping of the observations, or records, or rows in a dataset. This natural grouping should be in such a way that within groups, similarity should be at a maximum and between groups similarity should be at a minimum.

In real-world scenarios, there are cases where regression does not help predict the target variable. In supervised regression techniques, the input data is also known as *training data* . For each record, there is a label that has a continuous numerical value. The model is prepared through a training process that predicts the right output, and the process continues until the desired level of accuracy is achieved. We may need advanced regression methods to understand the pattern existing in the dataset.

## Introduction to Linear Regression

Linear regression analysis is known as the most reliable, easy to apply, and most widely used among all statistical techniques. This assumes linear, additive relationships between dependent and independent variables. The objective of linear regression is to predict the dependent or target variable through independent variables. The specification of the linear regression model is as follows.

Y = α + βX

This formula has a property in which the prediction for Y is a straight-line function of each of the X variables, keeping all others fixed, and the contributions of different X variables for the predictions are additive. The slopes of their individual straight-line relationships with Y are the coefficients of the variables. The coefficients and intercept are estimated by least squares (i.e., setting them equal to the unique values that minimize the sum of squared errors within the sample of data to which the model is fitted).

The model’s prediction errors are typically assumed to be independently and identically normally distributed. When the beta coefficient becomes zero, the input variable X has no impact on the dependent variable. The OLS method attempts to minimize the sum of the squared residuals. The residuals are defined as the difference between the points on the regression line to the actual data points in the scatterplot. This process seeks to estimate the beta coefficients in a multiple linear regression model.

Let’s take a sample dataset of 15 people. We capture the height and weight for each of them. By taking only their heights, can we predict the weight of a person using a linear regression technique? The answer is yes.

| **Person** | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Height** | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 |
| **Weight** | 115 | 117 | 120 | 123 | 126 | 129 | 132 | 135 | 139 | 142 | 146 | 150 | 154 | 159 | 164 |

To represent this graphically, we measure height on the x axis, and we measure weight on the y axis. The linear regression equation is on the graph where the intercept is 87.517 and the coefficient is 3.45\. The data points are represented by dots and the connecting line shows linear relationship (see Figure [5-2](#474315_1_En_5_Chapter.xhtml#Fig2)).

![../images/474315_1_En_5_Chapter/474315_1_En_5_Fig2_HTML.png](images/474315_1_En_5_Chapter/474315_1_En_5_Fig2_HTML.png)

Figure 5-2

Height and weight relationships

Why do we assume that a linear relationship exists between the dependent variable and a set of independent variables, when most o real-life scenarios reflect any other type of relationship than a linear relationship? The reasons why we stick to linear relationship are described next.

It is easy to understand and interpret. There are ways to transform an existing deviation from linearity and make it linear. It is simple to generate prediction.

The field of predictive modeling is mainly concerned with minimizing the errors in a predictive model, or making the most accurate predictions possible. Linear regression was developed in the field of statistics. It is studied as a model for understanding the relationship between the input and the output of numerical variables, but it has been borrowed by machine learning. It is both a statistical algorithm and a machine learning algorithm. The linear regression model depends on the following set of assumptions.

*   The linear relationship between dependent and independent variables.

*   There should not be any multicollinearity among the predictors. If we have more than two predictors in the input feature space, the input features should not be correlated.

*   There should not be any autocorrelation.

*   There should not be any heteroscedasticity. The variance of the error term should be constant, along the predictors on another axis, which means the error variance should be constant.

*   The error term should be normally distributed. The error term is basically defined as the difference between an actual and a predicted variable.

Within linear regression, there are different variants but in machine learning we consider them as one method. For example, if we are using one explanatory variable to predict the dependent variable, it is called a *simple linear regression model* . If we are using more than one explanatory variable, then the model is called a *multiple linear regression model* . The ordinary least square is a statistical technique to predict the linear regression model; hence, sometimes the linear regression model is also known as an *ordinary least square model* .

Linear regression is very sensitive to missing values and outliers because the statistical method of computing a linear regression depends on the mean, standard deviation, and covariance between the variables. Mean is sensitive to outlier values; therefore, it is expected that we need to clear out the outliers before proceeding toward forming the linear regression model.

In machine learning literature, the method for getting optimum beta coefficients that minimize the error in a regression model is achieved by a method called a *gradient descent algorithm* . How does the gradient descent algorithm work? It starts with an initial value, preferably from zero, and updates the scaling factor by a learning rate regularly iteratively to minimize the error term.

Understanding linear regression based on a machine learning approach requires special data preparation that avoids assumptions by keeping the original data intact. Data transformation is required to make your model more robust.

## Recipe 5-1\. Data Preparation for the Supervised Model

### Problem

How do we perform data preparation for creating a supervised learning model using PyTorch?

### Solution

We take an open source dataset, mtcars.csv, which is a regression dataset, to test how to create an input and output tensor.

### How It Works

First, the necessary library needs to be imported.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figa_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figa_HTML.jpg)

The predictor for the supervised algorithm is qsec, which is used to predict the mileage per gallon provided by the car. What is important here is the data type. First, we import the data, which is in NumPy format, into a PyTorch tensor format. The default tensor format is a float. Using the tensor float format would cause errors when performing the optimization function, so it is important to change the tensor data type. We can reformat the tensor type by using the unsqueeze function and specifying that the dimension is equal to 1.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figb_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figb_HTML.jpg)

To reproduce the same result, a manual seed needs to be set; so torch.manual_seed(1234) was used. Although we see that the data type is a tensor, if we check the type function, it will show as double, because a tensor type double is required for the optimization function.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figc_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figc_HTML.jpg)

## Recipe 5-2\. Forward and Backward Propagation

### Problem

How do we build a neural network torch class function so that we can build a forward propagation method?

### Solution

Design the neural network class function, including the hidden layer from the input layer and from the hidden layer to the output layer. In the neural network architecture, the number of neurons in the hidden layer also needs to be specified.

### How It Works

In the class Net() function, we first initialize the feature, hidden, and output layers. Then we introduce the back-propagation function using the rectified linear unit as the activation function in the hidden layer.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figd_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figd_HTML.jpg)

The following image shows the ReLU activation function. It is popularly used across different neural network models; however, the choice of the activation function should be based on accuracy. If we get more accuracy in a sigmoid function, we should consider that.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Fige_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Fige_HTML.jpg)

Now the network architecture is mentioned in the supervised learning model. The n_feature shows the number of neurons in the input layer. Since we have one input variable, qsec, we will use 1\. The number of neurons in the hidden layer can be decided based on the input and the degree of accuracy required in the learning model. We use the n_hidden equal to 20, which means 20 neurons in the hidden layer 1, and the output neuron is 1.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figf_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figf_HTML.jpg)

The role of the optimization function is to minimize the loss function defined with respect to the parameters and the learning rate. The learning rate chosen here is 0.2\. We also pass the neural network parameters into the optimizer. There are various optimization functions.

*   *SGD* . Implements stochastic gradient descent (optionally with momentum). The parameters could be momentum, learning rate, and weight decay.

*   *Adadelta* . Adaptive learning rate. Has five different arguments, parameters of the network, a coefficient used for computing a running average of the squared gradients, the addition of a term for achieving numerical stability of the model, the learning rate, and a weight decay parameter to apply regularization.

*   *Adagrad* . Adaptive subgradient methods for online learning and stochastic optimization. Has arguments such as iterable of parameter to optimize the learning rate and learning rate decay with weight decay.

*   *Adam* . A method for stochastic optimization. This function has six different arguments, an iterable of parameters to optimize, learning rate, betas (known as coefficients used for computing running averages of the gradient and its square), a parameter to improve numerical stability, and so forth.

*   *ASGD* . Acceleration of stochastic approximation by averaging. It has five different arguments, iterable of parameters to optimize, learning rate, decay term, weight decay, and so forth.

*   *RMSprop algorithm* . Uses a magnitude of gradients that are calculated to normalize the gradients.

*   *SparseAdam* . Implements a lazy version of the Adam algorithm suitable for sparse tensors. In this variant, only moments that show up in the gradient are updated, and only those portions of the gradient are applied to the parameters.

Apart from the optimization function, a loss function needs to be selected before running the supervised learning model. Again, there are various loss functions; let’s look at the error functions.

*   *MSELoss*. Creates a criterion that measures the mean squared error between elements in the input variable and target variable. For regression-related problems, this is the best loss function.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figg_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figg_HTML.jpg)

After running the supervised learning model, which is a regression model, we need to print the actual vs. predicted values and represent them in a graphical format; therefore, we need to turn on the interactive feature of the model.

## Recipe 5-3\. Optimization and Gradient Computation

### Problem

How do we build a basic supervised neural network training model using PyTorch with different iterations?

### Solution

The basic neural network model in PyTorch requires six different steps: preparing training data, initializing weights, creating a basic network model, calculating loss function, selecting the learning rate, and optimizing the loss function with respect to the parameters of the model.

### How It Works

Let’s follow a step-by-step approach to create a basic neural network model.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figh_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figh_HTML.jpg)

The final prediction result from the model with the first iteration and the last iteration is now represented in the following graph.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figi_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figi_HTML.jpg)

In the initial step, the loss function was 276.91. After optimization, the loss function became 35.1890\. The fitted regression line and the way it is fitted to the dataset are represented.

## Recipe 5-4\. Viewing Predictions

### Problem

How do we extract the best results from the PyTorch-based supervised learning model?

### Solution

The computational graph network is represented by nodes and connected through functions. Various techniques can be applied to minimize the error function and get the best predictive model. We can increase the iteration numbers, estimate the loss function, optimize the function, print actual and predicted values, and show it in a graph.

### How It Works

To apply tensor differentiation, the nn.backward() method needs to be applied. Let’s take an example to see how the error gradients are backpropagated. The grad() function holds the final output from the tensor differentiation.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figj_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figj_HTML.jpg)

The tuning parameters that can increase the accuracy of the supervised learning model, which is a regression use case, can be achieved with the following methods.

*   Number of iterations

*   Type of loss function

*   Selection of optimization method

*   Selection of loss function

*   Learning rate

*   Decay in the learning rate

*   Momentum require for optimization

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figk_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figk_HTML.jpg)

The real dataset looks like the following.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figl_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figl_HTML.jpg)

The following script explains reading the mpg and qsec columns from the mtcars.csv dataset. It converts those two variables to tensors using the unsqueeze function, and then uses it inside the neural network model for prediction.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figm_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figm_HTML.jpg)

After 1000 iterations, the model converges.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Fign_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Fign_HTML.jpg)

The neural networks in the torch library are typically used with the nn module. Let’s take a look at that.

Neural networks can be constructed using the torch.nn package, which provides almost all neural network related functionalities, including the following.

*   *Linear layers* : nn.Linear, nn.Bilinear

*   *Convolution layers* : nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d

*   *Nonlinearities* : nn.Sigmoid, nn.Tanh, nn.ReLU, nn.LeakyReLU

*   *Pooling layers* : nn.MaxPool1d, nn.AveragePool2d

*   *Recurrent networks* : nn.LSTM, nn.GRU

*   *Normalization* : nn.BatchNorm2d

*   *Dropout* : nn.Dropout, nn.Dropout2d

*   *Embedding* : nn.Embedding

*   *Loss functions* : nn.MSELoss, nn.CrossEntropyLoss, nn.NLLLoss

The standard classification algorithm is another version of a supervised learning algorithm, in which the target column is a class variable and the features could be numeric and categorical.

## Recipe 5-5\. Supervised Model Logistic Regression

### Problem

How do we deploy a logistic regression model using PyTorch?

### Solution

The computational graph network is represented by nodes and connected through functions. Various techniques can be applied to minimize the error function and get the best predictive model. We can increase the iteration numbers, estimate the loss function, optimize the function, print actual and predicted values, and show it in a graph.

### How It Works

To apply tensor differentiation , the nn.backward() method needs to be applied. Let’s look at an example.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figo_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figo_HTML.jpg)

The following shows data preparation for a logistic regression model.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figp_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figp_HTML.jpg)

Let’s look at the sample dataset for classification.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figq_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figq_HTML.jpg)

Set up the neural network module for the logistic regression model.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figr_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figr_HTML.jpg)

Check the neural network configuration.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figs_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figs_HTML.jpg)

Run iterations and find the best solution for the sample graph.

![../images/474315_1_En_5_Chapter/474315_1_En_5_Figt_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Figt_HTML.jpg)

The first iteration provides almost 99% accuracy, and subsequently, the model provides 100% accuracy on the training data (see Figures [5-3](#474315_1_En_5_Chapter.xhtml#Fig3) and [5-4](#474315_1_En_5_Chapter.xhtml#Fig4)).

![../images/474315_1_En_5_Chapter/474315_1_En_5_Fig4_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Fig4_HTML.jpg)

Figure 5-4

Final accuracy

![../images/474315_1_En_5_Chapter/474315_1_En_5_Fig3_HTML.jpg](images/474315_1_En_5_Chapter/474315_1_En_5_Fig3_HTML.jpg)

Figure 5-3

Initial accuracy

Final accuracy shows 100, which is a clear case of overfitting, but we can control this by introducing the dropout rate, which is covered in the next chapter.

## Conclusion

This chapter discussed two major types of supervised learning algorithms—linear regression and logistic regression—and their implementation using sample datasets and the PyTorch program. Both algorithms are linear models, one for predicting real valued output and the other for separating one class from another class. Although we considered a two-class classification in the logistic regression example, it can be extended to a multiclass classification model.

# 6. Fine-Tuning Deep Learning Models Using PyTorch

Deep learning models are becoming very popular. They have very deep roots in the way biological neurons are connected and the way they transmit information from one node to another node in a network model.

Deep learning has a very specific usage, particularly when the single function–based machine learning techniques fail to approximate real-life challenges. For example, when the data dimension is very large (in the thousands), then standard machine learning algorithms fail to predict or classify the outcome variable. This is also not very efficient computationally. It consumes a lot of resources and model convergence never happens. Most prominent examples are object detection, image classification, and image segmentation.

The most commonly used deep learning algorithms can be classified into three groups.

*   *Convolutional neural network* . Mostly suitable for highly sparse datasets, image classification, image recognition, object detection, and so forth.

*   *Recurrent neural network* . Applicable to processing sequential information, if there is any internal sequential structure in the way data is generated. This includes music, natural language, audio, and video, where the information is consumed in a sequence.

*   *Deep neural network* . Typically applicable when a single layer of a machine learning algorithm cannot classify or predict correctly. There are three variants.
    *   *Deep network*, where the number of neurons present in each hidden layer is usually more than the previous layer

    *   *Wide network*, where the number of hidden layers are more than a usual neural network model

    *   *Both deep and wide network*, where the number of neurons and the number of layers in the network are very high

This chapter discusses how to fine-tune deep learning models using hyperparameters. There is a difference between the parameters and hyperparameters. Usually in the deep learning models, we are not interested in estimating the parameters because they are the weights and keep changing based on the initial values, learning rate, and number of iterations. What is important is deciding on the hyperparameters to fine-tune the models, as discussed in Chapter [3](#474315_1_En_3_Chapter.xhtml), so that optimum results can be derived.

## Recipe 6-1\. Building Sequential Neural Networks

### Problem

Is there any way to build sequential neural network models, as we do in Keras in PyTorch, instead of declaring the neural network models?

### Solution

If we declare the entire neural network model, line by line, with the number of neurons, number of hidden layers and iterations, choice of loss functions, optimization functions, and the selection of weight distribution, and so forth, it will be extremely cumbersome to scale the model. And, it is not foolproof—errors could crop up in the model. To avoid the issues in declaring the entire model line by line, we can use a high-level function that assumes certain default parameters in the back end and returns the result to the user with minimum hyperparameters. Yes, it is possible to not have to declare the neural network model.

### How It Works

Let’s look at how to create such models. In the Torch library, the neural network module contains a functional API (application programming interface) that contains various activation functions, as discussed in earlier chapters.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figa_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figa_HTML.jpg)

In the following lines of script, we create a simple neural network model with linear function as the activation function for input to the hidden layer, and the hidden layer to the output layer.

The following function requires declaring `class Net` , declaring the features, hidden neurons, and activation functions, which can be easily replaced by the sequential module.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figb_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figb_HTML.jpg)

Instead of using this script, we can change the class function and replace it with the sequential function. The Keras functions replace the TensorFlow functions, which means that many lines of TensorFlow code can be replaced by a few lines of Keras script. The same thing is possible in PyTorch without requiring any external modules. As an example, in the following, net2 explains the sequential model and net1 explains the preceding script. From a readability perspective, net2 is much better than net1.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figc_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figc_HTML.jpg)

If we simply print both the net1 and net2 model architectures, it does the same thing.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figd_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figd_HTML.jpg)

## Recipe 6-2\. Deciding the Batch Size

### Problem

How do we perform batch data training for a deep learning model using PyTorch?

### Solution

Training a deep learning model requires a large amount of labeled data. Typically, it is the process of finding a set of weights and biases in such a way that the loss function becomes minimal with respect to matching the target label. If the training process approximates well to the function, the prediction or classification becomes robust.

### How It Works

There are two methods for training a deep learning network: batch training and online training. The choice of training algorithm dictates the method of learning. If the algorithm is backpropagation, then online learning is better. For a deep and wide network model with various layers of backpropagation and forward propagation, then batch training is better.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Fige_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Fige_HTML.jpg)

In the training process, the batch size is 5; we can change the batch size to 8 and see the results. In online training process, the weights and biases are updated for every training example based on the variations between predicted result and actual result. However, in the batch training process, the differences between actual and predicted values which is error gets accumulated and computed as a single number over the batch size, and reported at the final layer.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figf_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figf_HTML.jpg)

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figg_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figg_HTML.jpg)

After training the dataset for five iterations, we can print the batch and step. If we compare the online training and batch training, batch training has many more advantages than online training. When the requirement is to train a huge dataset, there are memory constraints. When we cannot process a huge dataset in a CPU environment, batch training comes to the rescue. In a CPU environment, we can process large amounts of data with a smaller batch size.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figh_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figh_HTML.jpg)

We take the batch size as 8 and retrain the model.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figi_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figi_HTML.jpg)

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figj_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figj_HTML.jpg)

## Recipe 6-3\. Deciding the Learning Rate

### Problem

How do we identify the best solution based on learning rate and the number of epochs?

### Solution

We take a sample tensor and apply various alternative models and print model parameters. The learning rate and epoch number are associated with model accuracy. To reach the global minimum state of the loss function, it is important to keep the learning rate to a minimum and the epoch number to a maximum so that the iteration can take the loss function to the minimum state.

### How It Works

First, the necessary library needs to be imported. To find the minimum loss function, gradient descent is typically used as the optimization algorithm, which is an iterative process. The objective is to find the rate of decline of the loss function with respect to the trainable parameters.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figk_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figk_HTML.jpg)

The sample dataset taken for the experiment includes the following.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figl_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figl_HTML.jpg)

The sample dataset and the first five records would look like the following.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figm_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figm_HTML.jpg)

Using the PyTorch utility function, let’s load the tensor dataset, introduce the batch size, and test out.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Fign_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Fign_HTML.jpg)

Declare the neural network module.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figo_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figo_HTML.jpg)

Now, let’s look at the network architecture.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figp_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figp_HTML.jpg)

While performing the optimization, we can include many options; select the best among the best.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figq_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figq_HTML.jpg)

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figr_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figr_HTML.jpg)

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figs_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figs_HTML.jpg)

## Recipe 6-4\. Performing Parallel Training

### Problem

How do we perform parallel data training that includes a lot of models using PyTorch?

### Solution

The optimizers are really functions that augment the tensor. The process of finding a best model requires parallel training of many models. The choice of learning rate, batch size, and optimization algorithms make models unique and different from other models. The process of selecting the best model requires hyperparameter optimization.

### How It Works

First, the right library needs to be imported. The three hyperparameters (learning rate, batch size, and optimization algorithm) make it possible to train multiple models in parallel, and the best model is decided by the accuracy of the test dataset. The following script uses the stochastic gradient descent algorithm, momentum, RMS prop, and Adam as the optimization method.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figt_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figt_HTML.jpg)

Let’s look at the chart and epochs.

![../images/474315_1_En_6_Chapter/474315_1_En_6_Figu_HTML.jpg](images/474315_1_En_6_Chapter/474315_1_En_6_Figu_HTML.jpg)

## Conclusion

In this chapter, we looked at various ways to make the deep learning model learn from the training dataset. The training process can be made effective by using hyperparameters. The selection of the right hyperparameter is the key. The deep learning models (convolutional neural network, recurrent neural network, and deep neural network) are different in terms of architecture, but the training process and the hyperparameters remain the same. The choice of hyperparameters and selection process is much easier in PyTorch than any other framework.

# 7. Natural Language Processing Using PyTorch

Natural language processing is an important branch of computer science. It is the study of human language by computers performing various tasks. Natural language study is also known as *computational linguistics* . There are two different components of natural language processing: natural language understanding and natural language generation. *Natural language understanding* involves analysis and knowledge of the input language and responding to it. *Natural language generation* is the process of creating language from input text. Language can be used in various ways. One word may have different meanings, so removing ambiguity is an important part of natural language understanding.

The ambiguity level can be of three types.

*   *Lexical ambiguity* is based on parts of speech; deciding whether a word is a noun, verb, adverb, and so forth.

*   *Syntactic ambiguity* is where one sentence can have multiple interpretations; the subject and predicate are neutral.

*   *Referential ambiguity* is related to an event or scenario expressed in words.

Text analysis is a precursor to natural language processing and understanding. Text analysis means corpus creation creating a collected set of documents, and then removing white spaces, punctuation, stop words, junk values such as symbols, emojis, and so forth, which have no textual meaning. After clean up, the net task is to represent the text in vector form. This is done using the standard Word2vec model, or it can be represented in term frequency and inverse document frequency format (tf-idf). In today’s world, we see a lot of applications that use natural language processing; the following are some examples.

*   Spell checking applications—online and on smartphones. The user types a particular word and the system checks the meaning of the word and suggests whether the spelling needs to be corrected.

*   Keyword search has been an integral part of our lives over the last decade. Whenever we go to a restaurant, buy something, or visit some place, we do an online search. If the keyword typed is wrong, no match is retrieved; however, the search engine systems are so intelligent that they predict the user’s intent and suggest pages that user actually wants to search.

*   Predictive text is used in various chat applications. The user types a word, and based on the user’s writing pattern, a choice of next words appear. The user is prompted to select any word from the list to frame his sentence.

*   Question-and-answering systems like Google Home, Amazon Alexa, and so forth, allow users to interact with the system in natural language. The system processes that information, does an intelligent search, and retrieves the best results for the user.

*   Alternate data extraction is when actual data is not available to the user, but the user can use the Internet to fetch data that is publicly available, and search for relevant information. For example, if I want to buy a laptop, I want to compare the price of the laptop on various online portals. I have one system scrape the price information from various websites and provide a summary of the prices to me. This process is called *alternate data collection* using web scraping, text processing and natural language processing.

*   Sentiment analysis is a process of analyzing the mood of the customer, user, or agent from the text that they express. Customer reviews, movie reviews, and so forth. The text presented needs to be analyzed and tagged as a positive sentiment or a negative sentiment. Similar applications can be built using sentiment analysis.

*   Topic modeling is the process of finding distinct topics presented in the corpus. For example, if we take text from science, math, English, and biology, and jumble all the text, then ask the machine to classify the text and tell us how many topics exist in the corpus, and the machine correctly separates the words present in English from biology, biology from science, and so on so forth. This is called a perfect topic modeling system.

*   Text summarization is the process of summarizing the text from the corpus in a shorter format. If we have a two-page document that is 1000 words, and we need to summarize it in a 200-word paragraph, then we can achieve that by using text summarization algorithms.

*   Language translation is translating one language to another, such as English to French, French to German, and so on so forth. Language translation helps the user understand another language and make the communication process effective.

The study of human language is discrete and very complex. The same sentence may have many meanings, but it is specifically constructed for an intended audience. To understand the complexity of natural language, we not only need tools and programs but also the system and methods. The following five-step approach is followed in natural language processing to understand the text from the user.

*   Lexical analysis identifies the structure of the word.

*   Syntactic analysis is the study of English grammar and syntax.

*   Semantic analysis is the meaning of a word in a context.

*   PoS (point of sale) analysis is the understanding and parsing parts of speech.

*   Pragmatic analysis is understanding the real meaning of a word in context.

In this chapter, we use PyTorch to implement the steps that are most commonly used in natural language processing tasks.

## Recipe 7-1\. Word Embedding

### Problem

How do we create a word-embedding model using PyTorch?

### Solution

Word embedding is the process of representing the words, phrases, and tokens in a meaningful way in a vector structure. The input text is mapped to vectors of real numbers; hence, feature vectors can be used for further computation by machine learning or deep learning models.

### How It Works

The words and phrases are represented in real vector format. The words or phrases that have similar meanings in a paragraph or document have similar vector representation. This makes the computation process effective in finding similar words. There are various algorithms for creating embedded vectors from text. Word2vec and GloVe are known frameworks to execute word embeddings. Let’s look at the following example.

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figa_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figa_HTML.jpg)

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figb_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figb_HTML.jpg)

The following sets up an embedding layer.

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figc_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figc_HTML.jpg)

Let’s look at the sample text. The following text has two paragraphs, and each paragraph has several sentences. If we apply word embedding on these two paragraphs, then we will get real vectors as features from the text. Those features can be used for further computation.

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figd_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figd_HTML.jpg)

Tokenization is the process of splitting sentences into small chunks of tokens, known as *n-grams*. This is called a *unigram* if it is a single word, a *bigram* if it is two words, a *trigram* if it is three words, so on and so forth.

![../images/474315_1_En_7_Chapter/474315_1_En_7_Fige_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Fige_HTML.jpg)

The PyTorch n-gram language modeler can extract relevant key words.

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figf_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figf_HTML.jpg)

The n-gram extractor has three arguments: the length of the vocabulary to extract, a dimension of embedding vector, and context size. Let’s look at the loss function and the model specification.

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figg_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figg_HTML.jpg)

Apply the Adam optimizer.

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figh_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figh_HTML.jpg)

Context extraction from sentences is also important. Let’s look at the following function.

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figi_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figi_HTML.jpg)

## Recipe 7-2\. CBOW Model in PyTorch

### Problem

How do we create a CBOW model using PyTorch?

### Solution

There are two different methods to represent words and phrases in vectors: *continuous bag of words* (CBOW) and *skip gram* . The bag-of-words approach learns embedding vectors by predicting the word or phrase in context. Context means the words before and after the current word. If we take a context of size 4, this implies that the four words to the left of the current word and the four words to the right of it are considered for context. The model tries to find those eight words in another sentence to predict the current word.

### How It Works

Let’s look at the following example.

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figj_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figj_HTML.jpg)

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figk_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figk_HTML.jpg)

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figl_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figl_HTML.jpg)

Graphically, the bag-of-words model looks like what is shown in Figure [7-1](#474315_1_En_7_Chapter.xhtml#Fig1). It has three layers: input, which are the embedding vectors that take the words and phrases into account; the output vector, which is the relevant word predicted by the model; and the projection layer, which is a computational layer provided by the neural network model.

![../images/474315_1_En_7_Chapter/474315_1_En_7_Fig1_HTML.png](images/474315_1_En_7_Chapter/474315_1_En_7_Fig1_HTML.png)

Figure 7-1

CBOW model representation

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figm_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figm_HTML.jpg)

![../images/474315_1_En_7_Chapter/474315_1_En_7_Fign_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Fign_HTML.jpg)

## Recipe 7-3\. LSTM Model

### Problem

How do we create a LSTM model using PyTorch?

### Solution

The *long short-term memory* (LSTM) model, also known as the *specific form of recurrent neural network* model, is commonly used in the natural language processing field. Text and sentences come in sequences to make a meaningful sentence, so we need a model that remembers the long and short sequences of text to predict a word or text.

### How It Works

Let’s look at the following example.

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figo_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figo_HTML.jpg)

Prepare a sequence of words as training data to form the LSTM network.

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figp_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figp_HTML.jpg)

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figq_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figq_HTML.jpg)

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figr_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figr_HTML.jpg)

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figs_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figs_HTML.jpg)

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figt_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figt_HTML.jpg)

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figu_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figu_HTML.jpg)

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figv_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figv_HTML.jpg)

![../images/474315_1_En_7_Chapter/474315_1_En_7_Figw_HTML.jpg](images/474315_1_En_7_Chapter/474315_1_En_7_Figw_HTML.jpg)

### Index

### A

Activation functions bilinear function definition hyperbolic tangent function leaky ReLU linear function log sigmoid transfer function PyTorch *vs* . TensorFlow ReLU sigmoid function visualization, shape of Adadelta Adagrad Adam optimizer Algorithmic classification Alternate data collection Artificial neural network (ANN) Autoencoders architecture clustering model encoded data, 3D plot features hyperbolic tangent function layer MNIST dataset torchvision library Autograd

### B

Bernoulli distribution Beta distribution Bilinear function Binomial distribution

### C

Central processing units (CPUs) Computational graph Computational linguistics Continuous bag of words (CBOW) example representation vectors embedding Continuous uniform distribution Convolutional neural network (CNN) architecture computational process hyperparameters loader functionality MNIST dataset net1 object pickle file format pooling layer prediction services predictions process restore function restore_net() function save function test accuracy training loss

### D, E

Data mining Deep learning models batch size batch training CPU environment loss function online training hyperparameters learning rate parallel data training sequential neural network Deep neural network (DNN) Discrete probability distribution Double exponential distribution

### F

Facebook’s artificial intelligence

### G

GloVe Gradient computation Gradient descent algorithm Graphics processing units (GPUs)

### H

Hyperbolic tangent function

### I, J

Implementation, deep learning Installation, PyTorch

### K

Keyword search application

### L

Language translation Laplacian distribution Leaky ReLU Lexical ambiguity Linear function Linear regression assumptions formula gradient descent algorithm height and weight mean, standard deviation and covariance multiple linear regression model OLS method ordinary least square model prediction errors predictive modeling simple linear regression model specification of Logistic regression model Log sigmoid transfer function Long short-term memory (LSTM) model Loss function backward() function epochs estimated parameters final loss value grad function hyperparameters initial value iteration level learning rate linear equation computation mean square error (MSE) MSELoss parameter grid weight tensor

### M

Machine learning Mean computation Multidimensional tensor Multinomial distribution Multiple linear regression model Multiprocessing

### N

Natural language generation Natural language processing applications five-step approach Natural language understanding Network architecture Neural network (NN) activation ( *see* Activation functions) architecture data mining data preparation definition design error functions functionalities median, mode and standard deviation module Net() function network architecture optimization functions Adadelta Adagrad Adam ASGD RMSprop algorithm SGD SparseAdam ReLU activation function set_weight() function step-by-step approach structure tensor differentiation torch.nn package n-gram language modeler Normal distribution NumPy-based operations

### O

Optim module Optimization function Adadelta Adam backpropagation process epochs gradients loss function parameters predicted tensors regression line Tensor.backward() tensor values torch.no_grad() training set validation dataset Ordinary least square model

### P

Predictive text Probability distribution autoencoders ( *see* Autoencoders) CNN loss function ( *see* Loss function) math operations model overfitting dropout rate hyperparameters overfitting loss and dropout loss parameters predicted values training accuracy training dataset optimization function RNN types weights, dropout rate

### Q

Question-and-answering systems

### R

Rectified linear unit (ReLu) Recurrent neural network (RNN) Adam optimizer built-in functions dsets.MINIST() function embedding layers hyperparameters image dataset LSTM model memory network MNIST dataset predictions regression problems cos function nonlinear cyclical pattern output layer test accuracy test data time series weights Word2vec Referential ambiguity Regression learning RMSprop algorithm

### S

Sentiment analysis Sequential neural network class Net functional API hyperparameters Keras functions model architectures Sigmoid function Simple linear regression model Skip gram SparseAdam Standard deviation Statistical distribution Statistics Stochastic gradient descent (SGD) Stochastic variable Supervised learning computational graph network data preparation forward and backward propagation ( *see* Neural network (NN)) grad() function linear regression ( *see* Linear regression) logistic regression model methods mtcars.csv dataset nn.backward() method optimization and gradient computation training data Syntactic ambiguity

### T

Tensor differentiation TensorFlow functions Tensors arrange function clamp function data structure dimensions is_storage function is_tensor function logarithmic values LongTensor/index select function mathematical functions NumPy functions 1D split function transformation functions 2D unbind function uniform distribution Text analysis Text summarization Tokenization Topic modeling Training data

### U, V

Unsupervised learning Utility function Utils

### W, X, Y, Z

Weight initialization Word2vec Word embeddings context extraction defined example n-gram extractor vector format