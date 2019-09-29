# Data

Fetch data [here](https://projects.intra.42.fr/uploads/document/document/464/data.csv)

Here is the epurated specification :

```
1. Title: Wisconsin Diagnostic Breast Cancer (WDBC)

2. Source Information

Creators: 

	Dr. William H. Wolberg, General Surgery Dept., University of
	Wisconsin,  Clinical Sciences Center, Madison, WI 53792
	wolberg@eagle.surgery.wisc.edu

	W. Nick Street, Computer Sciences Dept., University of
	Wisconsin, 1210 West Dayton St., Madison, WI 53706
	street@cs.wisc.edu  608-262-6619

	Olvi L. Mangasarian, Computer Sciences Dept., University of
	Wisconsin, 1210 West Dayton St., Madison, WI 53706
	olvi@cs.wisc.edu 

Donor: Nick Street - November 1995

Results:

	- predicting field 2, diagnosis: B = benign, M = malignant
	- sets are linearly separable using all 30 input features
	- best predictive accuracy obtained using one separating plane
		in the 3-D space of Worst Area, Worst Smoothness and
		Mean Texture.  Estimated accuracy 97.5% using repeated
		10-fold crossvalidations.  Classifier has correctly
		diagnosed 176 consecutive new patients as of November
		1995. 

5. Number of instances: 569 

6. Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)

7. Attribute information

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

	a) radius (mean of distances from center to points on the perimeter)
	b) texture (standard deviation of gray-scale values)
	c) perimeter
	d) area
	e) smoothness (local variation in radius lengths)
	f) compactness (perimeter^2 / area - 1.0)
	g) concavity (severity of concave portions of the contour)
	h) concave points (number of concave portions of the contour)
	i) symmetry 
	j) fractal dimension ("coastline approximation" - 1)

Several of the papers listed above contain detailed descriptions of
how these features are computed. 

The mean, standard error, and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features.  For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

8. Missing attribute values: none

9. Class distribution: 357 benign, 212 malignant
```

In this project, the `f) compactness` could be is ignored because its a mix between two other features.
By removing this feature, the input of the mlp is decreased by one and the complexity is reduced.

# Sources
https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html
https://www.dbs.ifi.lmu.de/Lehre/MaschLernen/SS2014/Skript/Perceptron2014.pdf
https://blog.zhaytam.com/2018/08/15/implement-neural-network-backpropagation/
https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9
http://www.cristiandima.com/neural-networks-from-scratch-in-python/

## Optimisations

### Adaline— Adaptive Linear Neuron

https://medium.com/@benjamindavidfraser/understanding-basic-machine-learning-with-python-perceptrons-and-artificial-neurons-dfae8fe61700

# What number of layers and nodes should use the model ?

https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/

# Why dont we use logistic regression cost function into MLP ?

https://sebastianraschka.com/faq/docs/logisticregr-neuralnet.html

# What is the importance of the bias into the MLP ?

# What is an activation function ?

https://fr.wikipedia.org/wiki/Fonction_d%27activation

# Why is the softmax often used as activation function of output layer in MLP ?

https://datascience.stackexchange.com/questions/37357/why-is-the-softmax-function-often-used-as-activation-function-of-output-layer-in