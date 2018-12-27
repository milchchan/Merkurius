using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Megalopolis
{
    namespace Layers
    {
        public class Convolutional : Layer, IUpdatable
        {
            private double[] weights = null;
            private double[] biases = null;
            private int channels = 0;
            private int imageWidth = 0;
            private int imageHeight = 0;
            private int filters = 0;
            private int filterWidth = 0;
            private int filterHeight = 0;
            private Batch<double[]> internalInputs = null;
            private double[][] gradients = null;

            public double[] Weights
            {
                get
                {
                    return this.weights;
                }
                set
                {
                    this.weights = value;
                }
            }

            public double[] Biases
            {
                get
                {
                    return this.biases;
                }
                set
                {
                    this.biases = value;
                }
            }

            public Convolutional(int channels, int imageWidth, int imageHeight, int filters, int filterWidth, int filterHeight, int poolWidth, int poolHeight, Func<int, int, int, double> func) : base(channels * imageWidth * imageHeight, filters * (imageWidth - filterWidth + 1) * (imageHeight - filterHeight + 1))
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

                for (int i = 0; i < length1; i++)
                {
                    this.weights[i] = func(i, fanIn, fanOut);
                }

                for (int i = 0; i < length2; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public Convolutional(Layer layer, int channels, int imageWidth, int imageHeight, int filters, int filterWidth, int filterHeight, int poolWidth, int poolHeight, Func<int, int, int, double> func) : base(layer, filters * (imageWidth - filterWidth + 1) * (imageHeight - filterHeight + 1))
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

                for (int i = 0; i < length1; i++)
                {
                    this.weights[i] = func(i, fanIn, fanOut);
                }

                for (int i = 0; i < length2; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                this.internalInputs = inputs;

                return this.Convolve(inputs, GetActivationMapWidth(), GetActivationMapHeight());
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                var parallelOptions = new ParallelOptions();
                var activationMapWidth = GetActivationMapWidth();
                var activationMapHeight = GetActivationMapHeight();
                var length = this.filters * this.channels * this.filterWidth * this.filterHeight;

                this.gradients = new double[deltas.Size][];

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(deltas, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector, state, index, local) =>
                {
                    var gradients = new double[length];

                    for (int i = 0; i < length; i++)
                    {
                        gradients[i] = 0.0;
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
                                            gradients[o] += vector[j] * this.internalInputs[index][n + this.imageWidth * (k + p) + l + q];
                                            o++;
                                        }
                                    }
                                }

                                j++;
                            }
                        }
                    }

                    local.Add(Tuple.Create<long, double[]>(index, gradients.Concat<double>(vector).ToArray<double>()));

                    return local;
                }, (local) =>
                {
                    lock (this.gradients)
                    {
                        local.ForEach(x =>
                        {
                            this.gradients[x.Item1] = x.Item2;
                        });
                    }
                });

                return DerivativeOfConvolve(deltas, activationMapWidth, activationMapHeight);
            }

            public Batch<double[]> GetGradients()
            {
                return new Batch<double[]>(this.gradients);
            }

            public void Update(Batch<double[]> gradients, Func<double, double, double> func)
            {
                var length1 = this.filters * this.channels * this.filterWidth * this.filterHeight;
                var length2 = this.filters * GetActivationMapWidth() * GetActivationMapHeight();

                for (int i = 1; i < gradients.Size; i++)
                {
                    for (int j = 0; j < length1; j++)
                    {
                        gradients[0][j] += gradients[i][j];
                    }

                    for (int j = 0, k = length1; j < length2; j++, k++)
                    {
                        gradients[0][k] += gradients[i][k];
                    }
                }

                for (int i = 0; i < length1; i++)
                {
                    this.weights[i] = func(this.weights[i], gradients[0][i] / gradients.Size);
                }

                for (int i = 0, j = length1; i < length2; i++, j++)
                {
                    this.biases[i] = func(this.biases[i], gradients[0][j] / gradients.Size);
                }
            }

            private Batch<double[]> Convolve(Batch<double[]> inputs, int activationMapWidth, int activationMapHeight)
            {
                var parallelOptions = new ParallelOptions();
                var data = new double[inputs.Size][];

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;
                
                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(inputs, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector, state, index, local) =>
                {
                    var convolvedInputs = new double[this.filters, activationMapHeight, activationMapWidth];
                    var activationMaps = new double[this.filters * activationMapHeight * activationMapWidth];

                    for (int i = 0; i < this.filters; i++)
                    {
                        for (int j = 0; j < activationMapHeight; j++)
                        {
                            for (int k = 0; k < activationMapWidth; k++)
                            {
                                convolvedInputs[i, j, k] = 0.0;
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
                                            convolvedInputs[i, k, l] += vector[n + this.imageWidth * (k + p) + l + q] * this.weights[o];
                                            o++;
                                        }
                                    }
                                }

                                activationMaps[j] = convolvedInputs[i, k, l] + this.biases[j];
                                j++;
                            }
                        }
                    }

                    local.Add(Tuple.Create<long, double[]>(index, activationMaps));

                    return local;
                }, (local) =>
                {
                    lock (data)
                    {
                        local.ForEach(x =>
                        {
                            data[x.Item1] = x.Item2;
                        });
                    }
                });

                return new Batch<double[]>(data);
            }

            private Batch<double[]> DerivativeOfConvolve(Batch<double[]> deltas, int activationMapWidth, int activationMapHeight)
            {
                var parallelOptions = new ParallelOptions();
                var length = this.channels * this.imageHeight * this.imageWidth;
                var data = new double[deltas.Size][];

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(deltas, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector, state, index, local) =>
                {
                    var d = new double[length];

                    for (int i = 0; i < length; i++)
                    {
                        d[i] = 0.0;
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
                                                d[k] += vector[n * activationMapWidth * activationMapHeight + y * activationMapWidth + x] * this.weights[o + j + q + r];
                                            }
                                        }
                                    }
                                }

                                k++;
                            }
                        }
                    }

                    local.Add(Tuple.Create<long, double[]>(index, d));

                    return local;
                }, (local) =>
                {
                    lock (data)
                    {
                        local.ForEach(x =>
                        {
                            data[x.Item1] = x.Item2;
                        });
                    }
                });

                return new Batch<double[]>(data);
            }

            private int GetActivationMapWidth()
            {
                return this.imageWidth - this.filterWidth + 1;
            }

            private int GetActivationMapHeight()
            {
                return this.imageHeight - this.filterHeight + 1;
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
