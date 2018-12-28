# Megalopolis

This repository contains the portable deep learning (deep neural networks) library implementation for .NET platform. This library is written by C#.

## Installation

Megalopolis can install the ML.NET NuGet package from the .NET Core CLI command.

```sh
> dotnet add package Megalopolis
```

or from the NuGet package manager.

```sh
PM> Install-Package Megalopolis
```

## Build

To build Megalopolis, run .NET Core CLI command.

```sh
> dotnet build Megalopolis.csproj
```

## Example

Convolutional neural network (CNN).

```csharp
var inputLayer = new Convolutional(1, 28, 28, 30, 5, 5, (index, fanIn, fanOut) => Initializers.HeNormal(fanIn));
var activation1 = new Activation(inputLayer, new ReLU());
var poolingLayer = new MaxPooling(activation1, 30, inputLayer.ActivationMapWidth, inputLayer.ActivationMapHeight, 2, 2);
var hiddenLayer = new FullyConnected(poolingLayer, 100, (index, fanIn, fanOut) => Initializers.HeNormal(fanIn));
var activation2 = new Activation(hiddenLayer, new ReLU());
var outputLayer = new Softmax(activation2, 10, (index, fanIn, fanOut) => Initializers.GlorotNormal(fanIn, fanOut));
var model = new Model(outputLayer, new Adam(), new SoftmaxCrossEntropy());

model.Fit(trainingList, 50);
```

## Features

* .NET Standard 2.0 library
* Code first modeling
* Dependency-free

### Activation Functions
* ELU (Exponential linear unit)
* Hyperbolic tangent
* Identity
* ReLU (Rectified linear unit)
* SELU (Scaled exponential linear unit)
* Sigmoid
* Softmax
* SoftPlus
* Softsign

### Layers
* Batch normalization
* Convolutional
* Dropout
* Embedding
* GLU (Gated recurrent unit)
* Fully connected
* LSTM (Long short-term memory)
* Max pooling
* Recurrent

### Loss Functions
* Cross-entropy
* Mean squared error (MSE)

### Optimizers
* AdaDelta
* AdaGrad
* Adam
* Momentum
* Nesterov
* RMSprop
* SGD
