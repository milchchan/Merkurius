using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Megalopolis
{
    namespace Layers
    {
        public class MaxPooling : Layer
        {
            private int filters = 0;
            private int activationMapWidth = 0;
            private int activationMapHeight = 0;
            private int poolWidth = 0;
            private int poolHeight = 0;
            private Batch<double[]> activationMaps = null;
            private Batch<double[]> internalOutputs = null;

            public MaxPooling(int filters, int activationMapWidth, int activationMapHeight, int poolWidth, int poolHeight) : base(filters * activationMapWidth * activationMapHeight, filters * activationMapWidth / poolWidth * (activationMapHeight / poolHeight))
            {
                this.filters = filters;
                this.activationMapWidth = activationMapWidth;
                this.activationMapHeight = activationMapHeight;
                this.poolWidth = poolWidth;
                this.poolHeight = poolHeight;
            }

            public MaxPooling(Layer layer, int filters, int activationMapWidth, int activationMapHeight, int poolWidth, int poolHeight) : base(layer, filters * activationMapWidth / poolWidth * (activationMapHeight / poolHeight))
            {
                this.filters = filters;
                this.activationMapWidth = activationMapWidth;
                this.activationMapHeight = activationMapHeight;
                this.poolWidth = poolWidth;
                this.poolHeight = poolHeight;
            }

            public MaxPooling(int filters, int activationMapWidth, int activationMapHeight, int poolWidth, int poolHeight, Layer layer) : base(filters * activationMapWidth * activationMapHeight, layer)
            {
                this.filters = filters;
                this.activationMapWidth = activationMapWidth;
                this.activationMapHeight = activationMapHeight;
                this.poolWidth = poolWidth;
                this.poolHeight = poolHeight;
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                this.activationMaps = inputs;
                this.internalOutputs = Pooling(inputs, this.activationMapWidth, this.activationMapHeight, GetOutputWidth(this.activationMapWidth), GetOutputHeight(this.activationMapHeight));

                return this.internalOutputs;
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                return DerivativeOfPooling(this.internalOutputs, deltas, this.activationMapWidth, this.activationMapHeight, GetOutputWidth(this.activationMapWidth), GetOutputHeight(this.activationMapHeight));
            }

            private Batch<double[]> Pooling(Batch<double[]> inputs, int activationMapWidth, int activationMapHeight, int outputWidth, int outputHeight)
            {
                var parallelOptions = new ParallelOptions();
                var length = this.filters * outputWidth * outputHeight;
                var data = new double[inputs.Size][];

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(inputs, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector, state, index, local) =>
                {
                    var activations = new double[length];

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
                                        var o = i * activationMapWidth * activationMapHeight + y * activationMapWidth + x;

                                        if (max < vector[o])
                                        {
                                            max = vector[o];
                                        }
                                    }
                                }

                                activations[j] = max;
                                j++;
                            }
                        }
                    }

                    local.Add(Tuple.Create<long, double[]>(index, activations));

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

            private Batch<double[]> DerivativeOfPooling(Batch<double[]> outputs, Batch<double[]> deltas, int activationMapWidth, int activationMapHeight, int outputWidth, int outputHeight)
            {
                var parallelOptions = new ParallelOptions();
                var data = new double[deltas.Size][];

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(deltas, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector, state, index, local) =>
                {
                    var d = new double[this.filters * activationMapHeight * activationMapWidth];

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
                                        var o = i * activationMapWidth * activationMapHeight + y * activationMapWidth + x;

                                        if (outputs[index][j] == this.activationMaps[index][o])
                                        {
                                            d[o] = vector[j];
                                        }
                                        else
                                        {
                                            d[o] = 0.0;
                                        }
                                    }
                                }

                                j++;
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

            private int GetOutputWidth(int activationMapWidth)
            {
                return activationMapWidth / this.poolWidth;
            }

            private int GetOutputHeight(int activationMapHeight)
            {
                return activationMapHeight / this.poolHeight;
            }

            public static int GetOutputLength(int activationMapLength, int poolLength)
            {
                return activationMapLength / poolLength;
            }
        }
    }
}
