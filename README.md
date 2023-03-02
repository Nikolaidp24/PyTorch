# PyTorch

**In this Repo, I will demonstrate the usage of PyTorch in different categories of data types, namely audio, computer vision, and so on.**

## Example One: Rice Classification
In this example, I have created a script to classify different categories of rice types, there are in total 75000 different images of 5 different kind of rice. Source can be found cited in the citation document in the submodule.

After 3 epochs of batch_size=128 training and testing, I managed to bring the accuracy up to 97%, and saved the model's state dictionary in the same folder in case you want to try it out for yourself.


## Example Two: Animals-10 Classification
In ths example, I will be using CNN based on nn.Module and nn.Conv2d with ReLU() to build a simple first baseline model without image augmentation to see the initial performance of it. 

In the meantime I will be writing some simple functions to automate the following:
* Training and testing steps
* Model training as a whole
* Plot multiple loss/accuracy curves in the same graph for multiple models for easier side-by-side comparison

Then I will keep optimizing the current model by tweaking various features, namely learning rate/activation function/data augmentation to see how high of an accuracy score I can be achieving with my own model.

At last, I will be using transfer learning (**resnet18**) to see what score can be achieved using this pre-trained model and plot out the respective confusion matrix.
