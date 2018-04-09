using System;
using System.Collections.Generic;
using Megalopolis.ActivationFunctions;

namespace Megalopolis
{
    namespace Layers
    {
        public class ConvolutionalPoolingLayer : Layer
        {
            private int channels = 0;
            private int imageWidth = 0;
            private int imageHeight = 0;
            private int filters = 0;
            private int filterWidth = 0;
            private int filterHeight = 0;
            private int poolWidth = 0;
            private int poolHeight = 0;
            private double[,,] convolvedInputs = null;
            private double[,,] activationMaps = null;
            private IActivationFunction activationFunction = null;

            public IActivationFunction ActivationFunction
            {
                get
                {
                    return this.activationFunction;
                }
            }

            public ConvolutionalPoolingLayer(int channels, int imageWidth, int imageHeight, int filters, int filterWidth, int filterHeight, int poolWidth, int poolHeight, IActivationFunction activationFunction, Func<int, int, int, double> func, Layer layer) : base(channels * imageWidth * imageHeight, layer)
            {
                var activationMapWidth = imageWidth - filterWidth + 1;
                var activationMapHeight = imageHeight - filterHeight + 1;
                var length1 = filters * channels * filterWidth * filterHeight;
                var length2 = filters * activationMapWidth * activationMapHeight;
                var fanIn = channels * filterWidth * filterHeight;
                var fanOut = filters * filterWidth * filterHeight / (poolWidth * poolHeight);

                this.weights = new double[length1];
                this.biases = new double[length2];
                this.channels = channels;
                this.imageWidth = imageWidth;
                this.imageHeight = imageHeight;
                this.filters = filters;
                this.filterWidth = filterWidth;
                this.filterHeight = filterHeight;
                this.poolWidth = poolWidth;
                this.poolHeight = poolHeight;
                this.convolvedInputs = new double[this.filters, activationMapHeight, activationMapWidth];
                this.activationMaps = new double[this.filters, activationMapHeight, activationMapWidth];
                this.activationFunction = activationFunction;

                for (int i = 0; i < length1; i++)
                {
                    this.weights[i] = func(i, fanIn, fanOut);
                }

                for (int i = 0; i < length2; i++)
                {
                    this.biases[i] = 0;
                }
            }

            public ConvolutionalPoolingLayer(ConvolutionalPoolingLayer layer) : base(layer)
            {
                var activationMapWidth = layer.imageWidth - layer.filterWidth + 1;
                var activationMapHeight = layer.imageHeight - layer.filterHeight + 1;
                var length = layer.filters * layer.channels * layer.filterWidth * layer.filterHeight;

                this.weights = new double[length];
                this.biases = new double[layer.filters * activationMapWidth * activationMapHeight];
                this.imageWidth = layer.imageWidth;
                this.imageHeight = layer.imageHeight;
                this.channels = layer.channels;
                this.filters = layer.filters;
                this.filterWidth = layer.filterWidth;
                this.filterHeight = layer.filterHeight;
                this.poolWidth = layer.poolWidth;
                this.poolHeight = layer.poolHeight;
                this.convolvedInputs = new double[layer.filters, activationMapHeight, activationMapWidth];
                this.activationMaps = new double[layer.filters, activationMapHeight, activationMapWidth];
                this.activationFunction = layer.activationFunction;

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = layer.weights[i];
                }

                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < activationMapHeight; k++)
                    {
                        for (int l = 0; l < activationMapWidth; l++)
                        {
                            this.biases[j] = layer.biases[j];
                            this.convolvedInputs[i, k, l] = layer.convolvedInputs[i, k, l];
                            this.activationMaps[i, k, l] = layer.activationMaps[i, k, l];
                            j++;
                        }
                    }
                }
            }

            public ConvolutionalPoolingLayer(ConvolutionalPoolingLayer sourceLayer, Layer targetLayer) : base(sourceLayer, targetLayer)
            {
                var activationMapWidth = sourceLayer.imageWidth - sourceLayer.filterWidth + 1;
                var activationMapHeight = sourceLayer.imageHeight - sourceLayer.filterHeight + 1;
                var length = sourceLayer.filters * sourceLayer.channels * sourceLayer.filterWidth * sourceLayer.filterHeight;

                this.weights = new double[length];
                this.biases = new double[sourceLayer.filters * activationMapWidth * activationMapHeight];
                this.imageWidth = sourceLayer.imageWidth;
                this.imageHeight = sourceLayer.imageHeight;
                this.channels = sourceLayer.channels;
                this.filters = sourceLayer.filters;
                this.filterWidth = sourceLayer.filterWidth;
                this.filterHeight = sourceLayer.filterHeight;
                this.poolWidth = sourceLayer.poolWidth;
                this.poolHeight = sourceLayer.poolHeight;
                this.convolvedInputs = new double[sourceLayer.filters, activationMapHeight, activationMapWidth];
                this.activationMaps = new double[sourceLayer.filters, activationMapHeight, activationMapWidth];
                this.activationFunction = sourceLayer.activationFunction;

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = sourceLayer.weights[i];
                }

                for (int i = 0, j = 0; i < sourceLayer.filters; i++)
                {
                    for (int k = 0; k < activationMapHeight; k++)
                    {
                        for (int l = 0; l < activationMapWidth; l++)
                        {
                            this.biases[j] = sourceLayer.biases[j];
                            this.convolvedInputs[i, k, l] = sourceLayer.convolvedInputs[i, k, l];
                            this.activationMaps[i, k, l] = sourceLayer.activationMaps[i, k, l];
                            j++;
                        }
                    }
                }
            }

            public override void PropagateForward(bool isTraining)
            {
                var activationMapWidth = GetActivationMapWidth();
                var activationMapHeight = GetActivationMapHeight();

                MaxPooling(Convolve(activationMapWidth, activationMapHeight), GetOutputWidth(activationMapWidth), GetOutputHeight(activationMapHeight));
            }

            public override IEnumerable<double[]> PropagateBackward(ref double[] deltas, out double[] gradients)
            {
                var activationMapWidth = GetActivationMapWidth();
                var activationMapHeight = GetActivationMapHeight();
                var outputWidth = GetOutputWidth(activationMapWidth);
                var outputHeight = GetOutputHeight(activationMapHeight);
                var length = this.filters * this.channels * this.filterWidth * this.filterHeight;
                var d = DerivativeOfMaxPooling(deltas, activationMapWidth, activationMapHeight, outputWidth, outputHeight);
                
                gradients = new double[length];
                deltas = new double[this.filters * activationMapWidth * activationMapHeight];

                for (int i = 0; i < length; i++)
                {
                    gradients[i] = 0;
                }

                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < activationMapHeight; k++)
                    {
                        for (int l = 0; l < activationMapWidth; l++)
                        {
                            deltas[j] = this.activationFunction.Derivative(this.convolvedInputs[i, k, l] + this.biases[j]) * d[i, k, l];

                            for (int m = 0, n = 0, o = this.channels * this.filterWidth * this.filterHeight * i; m < this.channels; m++, n += this.imageWidth * this.imageHeight)
                            {
                                for (int p = 0; p < this.filterHeight; p++)
                                {
                                    for (int q = 0; q < this.filterWidth; q++)
                                    {
                                        gradients[o] += deltas[j] * this.inputActivations[n + this.imageWidth * (k + p) + l + q];
                                        o++;
                                    }
                                }
                            }
                            
                            j++;
                        }
                    }
                }

                return new double[][] { DerivativeOfConvolve(d, activationMapWidth, activationMapHeight) };
            }

            public override void Update(double[] gradients, double[] deltas, Func<double, double, double> func)
            {
                var length1 = this.filters * this.channels * this.filterWidth * this.filterHeight;
                var length2 = this.filters * GetActivationMapWidth() * GetActivationMapHeight();

                for (int i = 0; i < length1; i++)
                {
                    this.weights[i] = func(this.weights[i], gradients[i]);
                }

                for (int i = 0; i < length2; i++)
                {
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

            private double[,,] Convolve(int activationMapWidth, int activationMapHeight)
            {
                for (int i = 0; i < this.filters; i++)
                {
                    for (int j = 0; j < activationMapHeight; j++)
                    {
                        for (int k = 0; k < activationMapWidth; k++)
                        {
                            this.convolvedInputs[i, j, k] = 0;
                        }
                    }
                }

                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < activationMapHeight; k++)
                    {
                        for (int l = 0; l < activationMapWidth; l++)
                        {
                            for (int m = 0, n = 0, o = this.channels * this.filterWidth * this.filterHeight * i; m < this.channels; m++, n += this.imageWidth * this.imageHeight)
                            {
                                for (int p = 0; p < this.filterHeight; p++)
                                {
                                    for (int q = 0; q < this.filterWidth; q++)
                                    {
                                        this.convolvedInputs[i, k, l] += this.inputActivations[n + this.imageWidth * (k + p) + l + q] * this.weights[o];
                                        o++;
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

            private double[] DerivativeOfConvolve(double[,,] deltas, int activationMapWidth, int activationMapHeight)
            {
                var length = this.channels * this.imageHeight * this.imageWidth;
                var d = new double[length];

                for (int i = 0; i < length; i++)
                {
                    d[i] = 0;
                }

                for (int i = 0, j = 0, k = 0; i < this.channels; i++, j += this.filterWidth * this.filterHeight)
                {
                    for (int l = 0; l < this.imageHeight; l++)
                    {
                        for (int m = 0; m < this.imageWidth; m++)
                        {
                            for (int n = 0, o = 0; n < this.filters; n++, o += this.channels * this.filterWidth * this.filterHeight)
                            {
                                for (int p = 0, q = 0; p < this.filterHeight; p++, q += this.filterWidth)
                                {
                                    for (int r = 0; r < this.filterWidth; r++)
                                    {
                                        var x = m - (this.filterWidth - 1) - r;
                                        var y = l - (this.filterHeight - 1) - p;

                                        if (y >= 0 && x >= 0)
                                        {
                                            d[k] += deltas[n, y, x] * this.activationFunction.Derivative(this.convolvedInputs[n, y, x] + this.biases[activationMapWidth * activationMapHeight * n + activationMapWidth * y + x]) * this.weights[o + j + q + r];
                                        }
                                    }
                                }
                            }

                            k++;
                        }
                    }
                }

                return d;
            }

            private void MaxPooling(double[,,] inputs, int outputWidth, int outputHeight)
            {
                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < outputHeight; k++)
                    {
                        for (int l = 0; l < outputWidth; l++)
                        {
                            var max = Double.MinValue;

                            for (int m = 0; m < this.poolHeight; m++)
                            {
                                for (int n = 0; n < this.poolWidth; n++)
                                {
                                    var x = this.poolWidth * l + n;
                                    var y = this.poolHeight * k + m;

                                    if (max < inputs[i, y, x])
                                    {
                                        max = inputs[i, y, x];
                                    }
                                }
                            }

                            this.outputActivations[j] = max;
                            j++;
                        }
                    }
                }
            }

            private double[,,] DerivativeOfMaxPooling(double[] deltas, int activationMapWidth, int activationMapHeight, int outputWidth, int outputHeight)
            {
                var d = new double[this.filters, activationMapHeight, activationMapWidth];

                for (int i = 0, j = 0; i < this.filters; i++)
                {
                    for (int k = 0; k < outputHeight; k++)
                    {
                        for (int l = 0; l < outputWidth; l++)
                        {
                            for (int m = 0; m < this.poolHeight; m++)
                            {
                                for (int n = 0; n < this.poolWidth; n++)
                                {
                                    var x = this.poolWidth * l + n;
                                    var y = this.poolHeight * k + m;

                                    if (this.outputActivations[j] == this.activationMaps[i, y, x])
                                    {
                                        d[i, y, x] = deltas[j];
                                    }
                                    else
                                    {
                                        d[i, y, x] = 0;
                                    }
                                }
                            }

                            j++;
                        }
                    }
                }

                return d;
            }

            private int GetActivationMapWidth()
            {
                return this.imageWidth - this.filterWidth + 1;
            }

            private int GetActivationMapHeight()
            {
                return this.imageHeight - this.filterHeight + 1;
            }

            private int GetOutputWidth(int activationMapWidth)
            {
                return activationMapWidth / this.poolWidth;
            }

            private int GetOutputHeight(int activationMapHeight)
            {
                return activationMapHeight / this.poolHeight;
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
