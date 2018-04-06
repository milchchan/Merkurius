using System;
using System.Collections.Generic;
using Megalopolis.ActivationFunctions;

namespace Megalopolis
{
    namespace Layers
    {
        public class ConvolutionalPoolingLayer : Layer
        {
            private int imageWidth = 0;
            private int imageHeight = 0;
            private int channels = 0;
            private int filters = 0;
            private int filterWidth = 0;
            private int filterHeight = 0;
            private int poolWidth = 0;
            private int poolHeight = 0;
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

            public ConvolutionalPoolingLayer(int imageWidth, int imageHeight, int channels, int filters, int filterWidth, int filterHeight, int poolWidth, int poolHeight, IActivationFunction activationFunction, Func<int, int, int, double> func, Layer layer) : base(channels * imageWidth * imageHeight, layer)
            {
                var activationMapWidth = imageWidth - filterWidth + 1;
                var activationMapHeight = imageHeight - filterHeight + 1;
                var length = filters * activationMapWidth * activationMapHeight;
                var outputWidth = activationMapWidth / poolWidth;
                var outputHeight = activationMapHeight / poolHeight;
                var fanIn = channels * filterWidth * filterHeight;
                var fanOut = filters * filterWidth * filterHeight / (poolWidth * poolHeight);

                this.weights = new double[length];
                this.biases = new double[length];
                this.imageWidth = imageWidth;
                this.imageHeight = imageHeight;
                this.channels = channels;
                this.filters = filters;
                this.filterWidth = filterWidth;
                this.filterHeight = filterHeight;
                this.poolWidth = poolWidth;
                this.poolHeight = poolHeight;
                this.convolvedInputs = new double[this.filters, activationMapHeight, activationMapWidth];
                this.activationMaps = new double[this.filters, activationMapHeight, activationMapWidth];
                this.outputs = new double[this.filters, outputHeight, outputWidth];
                this.activationFunction = activationFunction;

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(i, fanIn, fanOut);
                    this.biases[i] = 0;
                }
            }

            public ConvolutionalPoolingLayer(ConvolutionalPoolingLayer layer) : base(layer)
            {
                var activationMapWidth = layer.imageWidth - layer.filterWidth + 1;
                var activationMapHeight = layer.imageHeight - layer.filterHeight + 1;
                var length = layer.filters * activationMapWidth * activationMapHeight;
                var outputWidth = activationMapWidth / layer.poolWidth;
                var outputHeight = activationMapHeight / layer.poolHeight;

                this.weights = new double[length];
                this.biases = new double[length];
                this.imageWidth = layer.imageWidth;
                this.imageHeight = layer.imageHeight;
                this.channels = layer.channels;
                this.filters = layer.filters;
                this.filterWidth = layer.filterWidth;
                this.filterHeight = layer.filterHeight;
                this.poolWidth = layer.poolWidth;
                this.poolHeight = layer.poolHeight;
                this.convolvedInputs = new double[this.filters, activationMapHeight, activationMapWidth];
                this.activationMaps = new double[this.filters, activationMapHeight, activationMapWidth];
                this.outputs = new double[this.filters, outputHeight, outputWidth];
                this.activationFunction = layer.activationFunction;

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = layer.weights[i];
                    this.biases[i] = layer.biases[i];
                }

                for (int i = 0; i < this.filters; i++)
                {
                    for (int j = 0; j < activationMapHeight; j++)
                    {
                        for (int k = 0; k < activationMapWidth; k++)
                        {
                            this.convolvedInputs[i, j, k] = layer.convolvedInputs[i, j, k];
                            this.activationMaps[i, j, k] = layer.activationMaps[i, j, k];
                        }
                    }
                }

                for (int i = 0; i < this.filters; i++)
                {
                    for (int j = 0; j < outputHeight; j++)
                    {
                        for (int k = 0; k < outputWidth; k++)
                        {
                            this.outputs[i, j, k] = layer.outputs[i, j, k];
                        }
                    }
                }
            }

            public ConvolutionalPoolingLayer(ConvolutionalPoolingLayer sourceLayer, Layer targetLayer) : base(sourceLayer, targetLayer)
            {
                var activationMapWidth = sourceLayer.imageWidth - sourceLayer.filterWidth + 1;
                var activationMapHeight = sourceLayer.imageHeight - sourceLayer.filterHeight + 1;
                var length = sourceLayer.filters * activationMapWidth * activationMapHeight;
                var outputWidth = activationMapWidth / sourceLayer.poolWidth;
                var outputHeight = activationMapHeight / sourceLayer.poolHeight;

                this.weights = new double[length];
                this.biases = new double[length];
                this.imageWidth = sourceLayer.imageWidth;
                this.imageHeight = sourceLayer.imageHeight;
                this.channels = sourceLayer.channels;
                this.filters = sourceLayer.filters;
                this.filterWidth = sourceLayer.filterWidth;
                this.filterHeight = sourceLayer.filterHeight;
                this.poolWidth = sourceLayer.poolWidth;
                this.poolHeight = sourceLayer.poolHeight;
                this.convolvedInputs = new double[this.filters, activationMapHeight, activationMapWidth];
                this.activationMaps = new double[this.filters, activationMapHeight, activationMapWidth];
                this.outputs = new double[this.filters, outputHeight, outputWidth];
                this.activationFunction = sourceLayer.activationFunction;

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = sourceLayer.weights[i];
                    this.biases[i] = sourceLayer.biases[i];
                }

                for (int i = 0; i < this.filters; i++)
                {
                    for (int j = 0; j < activationMapHeight; j++)
                    {
                        for (int k = 0; k < activationMapWidth; k++)
                        {
                            this.convolvedInputs[i, j, k] = sourceLayer.convolvedInputs[i, j, k];
                            this.activationMaps[i, j, k] = sourceLayer.activationMaps[i, j, k];
                        }
                    }
                }

                for (int i = 0; i < this.filters; i++)
                {
                    for (int j = 0; j < outputHeight; j++)
                    {
                        for (int k = 0; k < outputWidth; k++)
                        {
                            this.outputs[i, j, k] = sourceLayer.outputs[i, j, k];
                        }
                    }
                }
            }

            public override void PropagateForward(bool isTraining)
            {
                var unflattenInputs = new double[this.channels, this.imageHeight, this.imageWidth];

                for (int i = 0, j = 0; i < this.channels; i++)
                {
                    for (int k = 0; k < this.imageHeight; k++)
                    {
                        for (int l = 0; l < this.imageWidth; l++)
                        {
                            unflattenInputs[i, k, l] = this.inputActivations[j];
                            j++;
                        }
                    }
                }

                var unflattenOutputs = MaxPooling(Convolve(unflattenInputs));
                var outputWidth = GetOutputWidth(GetActivationMapWidth());
                var outputHeight = GetOutputHeight(GetActivationMapHeight());

                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < outputHeight; k++)
                    {
                        for (int l = 0; l < outputWidth; l++)
                        {
                            this.outputActivations[j] = unflattenOutputs[i, k, l];
                            j++;
                        }
                    }
                }
            }

            public override IEnumerable<double[]> PropagateBackward(ref double[] deltas, out double[] gradients)
            {
                var activationMapWidth = GetActivationMapWidth();
                var activationMapHeight = GetActivationMapHeight();
                var outputWidth = GetOutputWidth(activationMapWidth);
                var outputHeight = GetOutputHeight(activationMapHeight);
                var unflattenDeltas = new double[this.filters, outputHeight, outputWidth];
                
                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < outputHeight; k++)
                    {
                        for (int l = 0; l < outputWidth; l++)
                        {
                            unflattenDeltas[i, k, l] = deltas[j];
                            j++;
                        }
                    }
                }

                var d1 = DerivativeOfMaxPooling(unflattenDeltas, activationMapWidth, activationMapHeight);

                deltas = new double[this.filters * activationMapWidth * activationMapHeight];
                gradients = new double[this.filters * activationMapWidth * activationMapHeight];
                
                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < activationMapHeight; k++)
                    {
                        for (int l = 0; l < activationMapWidth; l++)
                        {
                            deltas[j] = this.activationFunction.Derivative(this.activationMaps[i, k, l]) * d1[i, k, l];
                            gradients[j] = deltas[j] * this.convolvedInputs[i, k, l];
                            j++;
                        }
                    }
                }

                var d2 = DerivativeOfConvolve(d1);
                var flattenDeltas = new double[this.channels * this.imageWidth * this.imageHeight];

                for (int i = 0, j = 0; i < this.channels; i++)
                {
                    for (int k = 0; k < this.imageHeight; k++)
                    {
                        for (int l = 0; l < this.imageWidth; l++)
                        {
                            flattenDeltas[j] = d2[i, k, l];
                            j++;
                        }
                    }
                }

                return new double[][] { flattenDeltas };
            }

            public override void Update(double[] gradients, double[] deltas, Func<double, double, double> func)
            {
                var length = this.filters * GetActivationMapWidth() * GetActivationMapHeight();

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(this.weights[i], gradients[i]);
                    this.biases[i] = func(this.biases[i], deltas[i]);
                }
            }

            public override Layer Copy()
            {
                return new ConvolutionalPoolingLayer(this);
            }

            public override Layer Copy(Layer layer)
            {
                return new ConvolutionalPoolingLayer(this, layer);
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

                for (int i = 0; i < this.filters; i++)
                {
                    for (int j = 0; j < activationMapHeight; j++)
                    {
                        for (int k = 0; k < activationMapWidth; k++)
                        {
                            this.convolvedInputs[i, j, k] = this.activationMaps[i, j, k] = 0;
                        }
                    }
                }

                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < activationMapHeight; k++)
                    {
                        for (int l = 0; l < activationMapWidth; l++)
                        {
                            for (int m = 0; m < this.channels; m++)
                            {
                                for (int n = 0; n < this.filterHeight; n++)
                                {
                                    for (int o = 0; o < this.filterWidth; o++)
                                    {
                                        this.convolvedInputs[i, k, l] += inputs[m, k + n, l + o];
                                    }
                                }
                            }

                            this.activationMaps[i, k, l] = this.activationFunction.Function(this.convolvedInputs[i, k, l] + this.biases[j]);
                            j++;
                        }
                    }
                }

                return this.activationMaps;
            }

            private double[,,] DerivativeOfConvolve(double[,,] deltas)
            {
                var activationMapWidth = GetActivationMapWidth();
                var activationMapHeight = GetActivationMapHeight();
                var d = new double[this.channels, this.imageHeight, this.imageWidth];

                for (int i = 0; i < this.channels; i++)
                {
                    for (int j = 0; j < this.imageHeight; j++)
                    {
                        for (int k = 0; k < this.imageWidth; k++)
                        {
                            d[i, j, k] = 0;
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
                                            d[i, j, k] += deltas[l, j - (this.filterHeight - 1) - m, k - (this.filterWidth - 1) - n] * this.activationFunction.Derivative(this.convolvedInputs[l, j - (this.filterHeight - 1) - m, k - (this.filterWidth - 1) - n]) * this.weights[activationMapWidth * activationMapHeight * l + activationMapWidth * (j - (this.filterHeight - 1) - m) + k - (this.filterWidth - 1) - n];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                return d;
            }

            private double[,,] MaxPooling(double[,,] inputs)
            {
                var outputWidth = GetOutputWidth(inputs.GetLength(1));
                var outputHeight = GetOutputHeight(inputs.GetLength(2));

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

                            this.outputs[i, j, k] = max;
                        }
                    }
                }

                return this.outputs;
            }

            private double[,,] DerivativeOfMaxPooling(double[,,] deltas, int activationMapWidth, int activationMapHeight)
            {
                var outputWidth = deltas.GetLength(1);
                var outputHeight = deltas.GetLength(2);
                var d = new double[this.filters, activationMapHeight, activationMapWidth];

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
                                        d[i, this.poolHeight * j + l, this.poolWidth * k + m] = deltas[i, j, k];
                                    }
                                    else
                                    {
                                        d[i, this.poolHeight * j + l, this.poolWidth * k + m] = 0;
                                    }
                                }
                            }
                        }
                    }
                }

                return d;
            }

            public static int GetOutputLength(int imageLength, int filterLength)
            {
                return imageLength - filterLength + 1;
            }

            public static int GetOutputLength(int imageLength, int filterLength, int poolLength)
            {
                return (imageLength - filterLength + 1) / poolLength;
            }

            public static double[] Flatten(double[,,] inputs, int channels, int imageWidth, int imageHeight)
            {
                double[] outputs = new double[channels * imageWidth * imageHeight];

                for (int i = 0, j = 0; i < channels; i++)
                {
                    for (int k = 0; k < imageHeight; k++)
                    {
                        for (int l = 0; l < imageWidth; l++)
                        {
                            outputs[j] = inputs[i, k, l];
                            j++;
                        }
                    }
                }

                return outputs;
            }
        }
    }
}
