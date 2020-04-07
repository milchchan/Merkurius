# Merkurius

This repository contains the portable deep learning (deep neural networks) library implementation for .NET platform. This library is written by C#.

![](https://github.com/kawatan/Merkurius/workflows/.NET%20Core/badge.svg)

## Installation

You can install the Merkurius NuGet package from the .NET Core CLI command.

```sh
> dotnet add package Merkurius
```

or from the NuGet package manager.

```sh
PM> Install-Package Merkurius
```

## Build

To build Merkurius, run .NET Core CLI command.

```sh
> dotnet build Merkurius.csproj
```

## Example

Convolutional neural network (CNN).

```csharp
var model = new Model(
  new Convolution(ch, iw, ih, f, fw, fh, (fanIn, fanOut) => Initializers.HeNormal(fanIn),
  new Activation(new ReLU(),
  new MaxPooling(f, mw, mh, pw, ph,
  new FullyConnected(f * ow * oh, (fanIn, fanOut) => Initializers.HeNormal(fanIn),
  new Activation(new ReLU(),
  new FullyConnected(100, (fanIn, fanOut) => Initializers.GlorotNormal(fanIn, fanOut),
  new Softmax(10))))))),
  new Adam(), new SoftmaxCrossEntropy());

model.Fit(trainingList, 50);
```

## Features

* .NET Standard 2.1 library
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
* Convolution
* Dropout
* Embedding
* GRU (Gated recurrent unit)
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
