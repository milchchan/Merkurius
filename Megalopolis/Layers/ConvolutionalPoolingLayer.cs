using System;
using System.Collections.Generic;
using Megalopolis.ActivationFunctions;

namespace Megalopolis
{
    namespace Layers
    {
        public class ConvolutionalPoolingLayer : Layer
        {
            private Random random = null;
            private int imageWidth = 0;
            private int imageHeight = 0;
            private int channels = 0;
            private int filters = 0;
            private int filterWidth = 0;
            private int filterHeight = 0;
            private int poolWidth = 0;
            private int poolHeight = 0;
            private double[,,] internalWeights = null;
            private double[,,] internalBiases = null;
            private double[,,] convolvedInputs = null;
            private double[,,] activationMaps = null;
            private double[,,] outputs = null;
            private IActivationFunction activationFunction = null;

            public IActivationFunction ActivationFunction
            {
                get
                {
                    return this.activationFunction;
                }
            }

            public ConvolutionalPoolingLayer(Random random, int nodes, int imageWidth, int imageHeight, int channels, int filters, int filterWidth, int filterHeight, int poolWidth, int poolHeight, IActivationFunction activationFunction) : base(channels * imageWidth * imageHeight)
            {
                var activationMapWidth = imageWidth - filterWidth + 1;
                var activationMapHeight = imageHeight - filterHeight + 1;
                var a = Math.Sqrt(6 / (channels * filterWidth * filterHeight + filters * filterWidth * filterHeight / (poolWidth * poolHeight)));

                this.random = random;
                this.imageWidth = imageWidth;
                this.imageHeight = imageHeight;
                this.channels = channels;
                this.filters = filters;
                this.filterWidth = filterWidth;
                this.filterHeight = filterHeight;
                this.poolWidth = poolWidth;
                this.poolHeight = poolHeight;
                this.internalWeights = new double[filters, activationMapHeight, activationMapWidth];
                this.internalBiases = new double[filters, activationMapHeight, activationMapWidth];
                this.activationFunction = activationFunction;

                for (int i = 0; i < filters; i++)
                {
                    for (int j = 0; j < activationMapHeight; j++)
                    {
                        for (int k = 0; k < activationMapWidth; k++)
                        {
                            this.internalWeights[i, j, k] = this.random.Uniform(-a, a);
                            this.internalBiases[i, j, k] = 0;
                        }
                    }
                }
            }

            public override void PropagateForward(bool isTraining)
            {
                var unflattenInputs = new double[this.channels, this.imageHeight, this.imageWidth];

                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < this.imageHeight; k++)
                    {
                        for (int l = 0; l < this.imageWidth; l++)
                        {
                            unflattenInputs[i, k, l] = this.activations[j];
                            j++;
                        }
                    }
                }

                var unflattenOutputs = MaxPooling(Convolve(unflattenInputs));
                var outputWidth = GetOutputWidth(GetActivationMapWidth());
                var outputHeight = GetOutputHeight(GetActivationMapHeight());
                var flattenOutputs = new double[this.filters * outputWidth * outputHeight];

                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < outputHeight; k++)
                    {
                        for (int l = 0; l < outputWidth; l++)
                        {
                            this.nextLayer.Activations[j] = flattenOutputs[j] = unflattenOutputs[i, k, l];
                            j++;
                        }
                    }
                }
            }

            public override double[] PropagateBackward(ref double[] gradients)
            {
                var activationMapWidth = GetActivationMapWidth();
                var activationMapHeight = GetActivationMapHeight();
                var outputWidth = GetOutputWidth(activationMapWidth);
                var outputHeight = GetOutputHeight(activationMapHeight);
                var unflattenGradients = new double[this.channels, this.imageHeight, this.imageWidth];

                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < outputHeight; k++)
                    {
                        for (int l = 0; l < outputWidth; l++)
                        {
                            unflattenGradients[i, k, l] = gradients[j];
                            j++;
                        }
                    }
                }

                var g1 = DerivativeOfMaxPooling(unflattenGradients, activationMapWidth, activationMapHeight);

                gradients = new double[this.filters * activationMapWidth * activationMapHeight];

                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < activationMapHeight; k++)
                    {
                        for (int l = 0; l < activationMapWidth; l++)
                        {
                            gradients[j] = g1[i, k, l];
                            j++;
                        }
                    }
                }

                var g2 = DerivativeOfConvolve(g1);
                var flattenGradients = new double[this.channels * this.imageWidth * this.imageHeight];

                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < outputHeight; k++)
                    {
                        for (int l = 0; l < outputWidth; l++)
                        {
                            flattenGradients[j] = g2[i, k, l];
                            j++;
                        }
                    }
                }

                return flattenGradients;
            }

            public override void Update(double[] gradients, Func<double, double, double> func)
            {
                var activationMapWidth = GetActivationMapWidth();
                var activationMapHeight = GetActivationMapHeight();
                var unflattenGradients = new double[this.filters, activationMapHeight, activationMapWidth];

                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < activationMapHeight; k++)
                    {
                        for (int l = 0; l < activationMapWidth; l++)
                        {
                            this.internalWeights[i, k, l] = func(this.internalWeights[i, k, l], gradients[j] * this.convolvedInputs[i, k, l]);
                            this.internalBiases[i, k, l] = func(this.internalBiases[i, k, l], gradients[j]);
                            j++;
                        }
                    }
                }
            }

            public int GetActivationMapWidth()
            {
                return this.imageWidth - this.filterWidth + 1;
            }

            public int GetActivationMapHeight()
            {
                return this.imageHeight - this.filterHeight + 1;
            }

            public int GetOutputWidth(int activationMapWidth)
            {
                return activationMapWidth / this.poolWidth;
            }

            public int GetOutputHeight(int activationMapHeight)
            {
                return activationMapHeight / this.poolHeight;
            }

            private double[,,] Convolve(double[,,] inputs)
            {
                var activationMapWidth = GetActivationMapWidth();
                var activationMapHeight = GetActivationMapHeight();
                var convolvedInputs = new double[this.filters, activationMapHeight, activationMapWidth];
                var activationMaps = new double[this.filters, activationMapHeight, activationMapWidth];

                for (int i = 0; i < this.filters; i++)
                {
                    for (int j = 0; j < activationMapHeight; j++)
                    {
                        for (int k = 0; k < activationMapWidth; k++)
                        {
                            convolvedInputs[i, j, k] = activationMaps[i, j, k] = 0;
                        }
                    }
                }

                for (int i = 0; i < this.filters; i++)
                {
                    for (int j = 0; j < activationMapHeight; j++)
                    {
                        for (int k = 0; k < activationMapWidth; k++)
                        {
                            for (int l = 0; l < this.channels; l++)
                            {
                                for (int m = 0; m < this.filterHeight; m++)
                                {
                                    for (int n = 0; n < this.filterWidth; n++)
                                    {
                                        convolvedInputs[i, j, k] += inputs[l, j + m, k + n];
                                    }
                                }
                            }

                            activationMaps[i, j, k] = this.activationFunction.Function(convolvedInputs[i, j, k] + this.internalBiases[i, j, k]);
                        }
                    }
                }

                this.convolvedInputs = convolvedInputs;
                this.activationMaps = activationMaps;

                return activationMaps;
            }

            private double[,,] DerivativeOfConvolve(double[,,] gradients)
            {
                var activationMapWidth = GetActivationMapWidth();
                var activationMapHeight = GetActivationMapHeight();
                var g = new double[this.channels, this.imageHeight, this.imageWidth];

                for (int i = 0; i < this.channels; i++)
                {
                    for (int j = 0; j < this.imageHeight; j++)
                    {
                        for (int k = 0; k < this.imageWidth; k++)
                        {
                            g[i, j, k] = 0;
                        }
                    }
                }

                for (int i = 0; i < this.channels; i++)
                {
                    for (int j = 0; j < this.imageHeight; j++)
                    {
                        for (int k = 0; k < this.imageWidth; k++)
                        {
                            for (int l = 0; l < this.filters; l++)
                            {
                                for (int m = 0; m < this.filterHeight; m++)
                                {
                                    for (int n = 0; n < this.filterWidth; n++)
                                    {
                                        if (j - (this.filterHeight - 1) - m >= 0 && k - (this.filterWidth - 1) - n >= 0)
                                        {
                                            g[i, j, k] += gradients[l, j - (this.filterHeight - 1) - m, k - (this.filterWidth - 1) - n] * this.activationFunction.Derivative(this.convolvedInputs[l, j - (this.filterHeight - 1) - m, k - (this.filterWidth - 1) - n]) * this.internalWeights[l, j - (this.filterHeight - 1) - m, k - (this.filterWidth - 1) - n];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                return g;
            }

            private double[,,] MaxPooling(double[,,] inputs)
            {
                var outputWidth = GetOutputWidth(inputs.GetLength(1));
                var outputHeight = GetOutputHeight(inputs.GetLength(2));
                var outputs = new double[this.filters, outputHeight, outputWidth];

                for (int i = 0; i < this.filters; i++)
                {
                    for (int j = 0; j < outputHeight; j++)
                    {
                        for (int k = 0; k < outputWidth; k++)
                        {
                            var max = Double.MinValue;

                            for (int l = 0; l < this.poolHeight; l++)
                            {
                                for (int m = 0; m < this.poolWidth; m++)
                                {
                                    if (max < inputs[i, this.poolHeight * j + l, this.poolWidth * k + m])
                                    {
                                        max = inputs[i, this.poolHeight * j + l, this.poolWidth * k + m];
                                    }
                                }
                            }

                            outputs[i, j, k] = max;
                        }
                    }
                }

                this.outputs = outputs;

                return outputs;
            }

            private double[,,] DerivativeOfMaxPooling(double[,,] gradients, int activationMapWidth, int activationMapHeight)
            {
                var outputWidth = gradients.GetLength(1);
                var outputHeight = gradients.GetLength(2);
                var g = new double[this.filters, activationMapHeight, activationMapWidth];

                for (int i = 0; i < this.filters; i++)
                {
                    for (int j = 0; j < outputHeight; j++)
                    {
                        for (int k = 0; k < outputWidth; k++)
                        {
                            for (int l = 0; l < this.poolHeight; l++)
                            {
                                for (int m = 0; m < this.poolWidth; m++)
                                {
                                    if (this.outputs[i, j, k] == this.activationMaps[i, this.poolHeight * j + l, this.poolWidth * k + m])
                                    {
                                        g[i, this.poolHeight * j + l, this.poolWidth * k + m] = gradients[i, j, k];
                                    }
                                    else
                                    {
                                        g[i, this.poolHeight * j + l, this.poolWidth * k + m] = 0;
                                    }
                                }
                            }
                        }
                    }
                }

                return g;
            }
        }
    }
}
