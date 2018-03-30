using System;
using System.Collections.Generic;
using Alice.ActivationFunctions;

namespace Alice
{
    namespace Layers
    {
        public class ConvolutionalPoolingLayer : Layer
        {
            private Random random = null;
            private int nodes = 0;
            private int imageWidth = 0;
            private int imageHeight = 0;
            private int channel = 0;
            private int numberOfKernel = 0;
            private int kernelWidth = 0;
            private int kernelHeight = 0;
            private int poolWidth = 0;
            private int poolHeight = 0;
            private double[,,,] _weights = null;
            private double[] _biases = null;
            private IActivationFunction activationFunction = null;

            public IActivationFunction ActivationFunction
            {
                get
                {
                    return this.activationFunction;
                }
            }

            public ConvolutionalPoolingLayer(Random random, int nodes, int imageWidth, int imageHeight, int channel, int numberOfKernel, int kernelWidth, int kernelHeight, int poolWidth, int poolHeight, IActivationFunction activationFunction) : base(nodes)
            {
                this.random = random;


                this.nodes = nodes;


                this.imageWidth = imageWidth;
                this.imageHeight = imageHeight;
                this.channel = channel;
                this.numberOfKernel = numberOfKernel;
                this.kernelWidth = kernelWidth;
                this.kernelHeight = kernelHeight;
                this.poolWidth = poolWidth;
                this.poolHeight = poolHeight;

                double a = Math.Sqrt(6 / (channel * kernelWidth * kernelHeight + numberOfKernel * kernelWidth * kernelHeight / (poolWidth * poolHeight)));

                this._weights = new double[numberOfKernel, channel, kernelWidth, kernelHeight];

                for (int i = 0; i < numberOfKernel; i++)
                {
                    for (int j = 0; j < channel; j++)
                    {
                        for (int k = 0; k < kernelWidth; k++)
                        {
                            for (int l = 0; l < kernelHeight; l++)
                            {
                                this._weights[i, j, k, l] = this.random.Uniform(-a, a);
                            }
                        }
                    }
                }

                this.biases = new double[numberOfKernel];

                for (int i = 0; i < numberOfKernel; i++)
                {
                    this._biases[i] = 0;
                }

                this.activationFunction = activationFunction;
            }

            public override void PropagateForward(bool isTraining)
            {
                throw new NotImplementedException();
            }

            public override double[] PropagateBackward(double[] gradients)
            {
                throw new NotImplementedException();
            }

            private void Convolve()
            {
                throw new NotImplementedException();
            }

            private double[] DerivativeOfConvolve(double[] x, int i)
            {
                throw new NotImplementedException();
            }

            private void MaxPooling()
            {
                throw new NotImplementedException();
            }

            private double[] DerivativeOfMaxPooling(double[] x, int i)
            {
                throw new NotImplementedException();
            }
        }
    }
}
