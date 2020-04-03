using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Threading.Tasks;
using Merkurius.ActivationFunctions;

namespace Merkurius
{
    namespace Layers
    {
        // Gated recurrent unit (GRU)
        [DataContract]
        public class GRU : Layer, IUpdatable
        {
            [DataMember]
            private double[] weights = null;
            [DataMember]
            private double[] biases = null;
            [DataMember]
            private int timesteps = 0;
            [DataMember]
            private bool stateful = false;
            private List<InternalGRU> layerList = null;
            private Batch<double[]> h = null;
            private Batch<double[]> dh = null;
            private double[][] gradients = null;
            [DataMember]
            private IActivationFunction tanhActivationFunction = null;
            [DataMember]
            private IActivationFunction sigmoidActivationFunction = null;

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

            public Batch<double[]> State
            {
                get
                {
                    return this.h;
                }
                set
                {
                    this.h = value;
                }
            }

            public GRU(int inputs, int outputs, Func<int, int, double> func) : base(inputs, outputs)
            {
                var length1 = outputs * 3;
                var length2 = inputs * length1 + outputs * length1;

                this.weights = new double[length2];
                this.biases = new double[length1];
                this.tanhActivationFunction = new HyperbolicTangent();
                this.sigmoidActivationFunction = new Sigmoid();

                for (int i = 0; i < length2; i++)
                {
                    this.weights[i] = func(inputs, outputs);
                }

                for (int i = 0; i < length1; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public GRU(Layer layer, int nodes, Func<int, int, double> func) : base(layer, nodes)
            {
                var length1 = outputs * 3;
                var length2 = layer.Outputs * length1 + nodes * length1;

                this.weights = new double[length2];
                this.biases = new double[length1];
                this.tanhActivationFunction = new HyperbolicTangent();
                this.sigmoidActivationFunction = new Sigmoid();

                for (int i = 0; i < length2; i++)
                {
                    this.weights[i] = func(layer.Outputs, nodes);
                }

                for (int i = 0; i < length1; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public GRU(int inputs, int hiddens, int timesteps, bool stateful, Func<int, int, double> func, Layer layer) : base(inputs, layer)
            {
                var length1 = hiddens * 3;
                var length2 = inputs * length1 + hiddens * length1;

                this.outputs = hiddens;
                this.weights = new double[length2];
                this.biases = new double[length1];
                this.timesteps = timesteps;
                this.stateful = stateful;
                this.tanhActivationFunction = new HyperbolicTangent();
                this.sigmoidActivationFunction = new Sigmoid();

                for (int i = 0; i < length2; i++)
                {
                    this.weights[i] = func(hiddens, inputs);
                }

                for (int i = 0; i < length1; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                var length1 = this.outputs * 3;
                var length2 = this.inputs * length1;
                var length3 = this.outputs * length1;
                var length4 = this.timesteps * this.outputs;
                var xWeights = new double[length2];
                var hWeights = new double[length3];
                var outputs = new double[inputs.Size][];

                for (int i = 0; i < length2; i++)
                {
                    xWeights[i] = this.weights[i];
                }

                for (int i = 0, j = length2; i < length3; i++, j++)
                {
                    hWeights[i] = this.weights[j];
                }

                for (int i = 0; i < inputs.Size; i++)
                {
                    outputs[i] = new double[length4];
                }

                this.layerList = new List<InternalGRU>();

                if (!this.stateful || this.h == null)
                {
                    this.h = new Batch<double[]>(new double[inputs.Size][]);

                    for (int i = 0; i < inputs.Size; i++)
                    {
                        this.h[i] = new double[this.outputs];

                        for (int j = 0; j < this.outputs; j++)
                        {
                            this.h[i][j] = 0.0;
                        }
                    }
                }

                for (int t = 0; t < this.timesteps; t++)
                {
                    var layer = new InternalGRU(this.inputs, this.outputs, xWeights, hWeights, this.biases, this.tanhActivationFunction, this.sigmoidActivationFunction);
                    var x = new Batch<double[]>(new double[inputs.Size][]);

                    for (int i = 0; i < inputs.Size; i++)
                    {
                        var vector = new double[this.inputs];

                        for (int j = 0, k = this.inputs * t; j < this.inputs; j++, k++)
                        {
                            vector[j] = inputs[i][k];
                        }

                        x[i] = vector;
                    }

                    this.h = layer.Forward(x, this.h);

                    for (int i = 0; i < inputs.Size; i++)
                    {
                        for (int j = 0, k = this.outputs * t; j < this.outputs; j++, k++)
                        {
                            outputs[i][k] = this.h[i][j];
                        }
                    }

                    this.layerList.Add(layer);
                }

                return new Batch<double[]>(outputs);
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                var length1 = this.outputs * 3;
                var length2 = this.timesteps * this.inputs;
                var length3 = this.inputs * length1 + this.outputs * length1 + length1;
                var d = new double[deltas.Size][];
                var dh = new Batch<double[]>(new double[deltas.Size][]);

                this.gradients = new double[deltas.Size][];

                for (int i = 0; i < deltas.Size; i++)
                {
                    d[i] = new double[length2];
                    dh[i] = new double[this.outputs];

                    for (int j = 0; j < this.outputs; j++)
                    {
                        dh[i][j] = 0.0;
                    }

                    this.gradients[i] = new double[length3];

                    for (int j = 0; j < length3; j++)
                    {
                        this.gradients[i][j] = 0.0;
                    }
                }

                for (int t = this.timesteps - 1; t >= 0; t--)
                {
                    for (int i = 0; i < deltas.Size; i++)
                    {
                        for (int j = 0, k = this.outputs * t; j < this.outputs; j++, k++)
                        {
                            dh[i][j] += deltas[i][k];
                        }
                    }

                    var tuple = this.layerList[t].Backward(dh);

                    dh = tuple.Item2;

                    for (int i = 0; i < deltas.Size; i++)
                    {
                        for (int j = 0, k = this.inputs * t; j < this.inputs; j++, k++)
                        {
                            d[i][k] = tuple.Item1[i][j];
                        }

                        for (int j = 0; j < length2; j++)
                        {
                            this.gradients[i][j] += tuple.Item3[i][j];
                        }
                    }
                }

                this.dh = dh;

                return new Batch<double[]>(d);
            }

            public Batch<double[]> GetGradients()
            {
                return new Batch<double[]>(this.gradients);
            }

            public void SetGradients(Func<bool, double, int, double> func)
            {
                var length = this.outputs * 3;
                var offset = this.inputs * length + this.outputs * length;

                foreach (double[] vector in this.gradients)
                {
                    for (int i = 0; i < offset; i++)
                    {
                        vector[i] = func(true, vector[i], i);
                    }

                    for (int i = 0, j = offset; i < length; i++, j++)
                    {
                        vector[j] = func(false, vector[j], i);
                    }
                }
            }

            public void Update(Batch<double[]> gradients, Func<double, double, double> func)
            {
                var length1 = this.outputs * 3;
                var length2 = this.inputs * length1;
                var length3 = this.outputs * length1;
                var offset = length2 + length3;

                for (int i = 1; i < gradients.Size; i++)
                {
                    for (int j = 0; j < length2; j++)
                    {
                        gradients[0][j] += gradients[i][j];
                    }

                    for (int j = 0, k = length2; j < length3; j++, k++)
                    {
                        gradients[0][k] += gradients[i][k];
                    }

                    for (int j = 0, k = offset; j < length1; j++, k++)
                    {
                        gradients[0][k] += gradients[i][k];
                    }
                }

                for (int i = 0; i < length2; i++)
                {
                    this.weights[i] = func(this.weights[i], gradients[0][i] / gradients.Size);
                }

                for (int i = 0, j = length2; i < length3; i++, j++)
                {
                    this.weights[j] = func(this.weights[j], gradients[0][j] / gradients.Size);
                }

                for (int i = 0, j = offset; i < length1; i++, j++)
                {
                    this.biases[i] = func(this.biases[i], gradients[0][j] / gradients.Size);
                }
            }

            private class InternalGRU
            {
                private int inputs = 0;
                private int hiddens = 0;
                private double[] xWeights = null;
                private double[] hWeights = null;
                private double[] biases = null;
                private IActivationFunction tanhActivationFunction = null;
                private IActivationFunction sigmoidActivationFunction = null;
                private Tuple<Batch<double[]>, Batch<double[]>, double[][], double[][], double[][]> cache = null;

                public InternalGRU(int inputs, int hiddens, double[] xWeights, double[] hWeights, double[] biases, IActivationFunction tanhActivationFunction, IActivationFunction sigmoidActivationFunction)
                {
                    this.inputs = inputs;
                    this.hiddens = hiddens;
                    this.xWeights = xWeights;
                    this.hWeights = hWeights;
                    this.biases = biases;
                    this.tanhActivationFunction = tanhActivationFunction;
                    this.sigmoidActivationFunction = sigmoidActivationFunction;
                }

                public Batch<double[]> Forward(Batch<double[]> x, Batch<double[]> hPrevious)
                {
                    var parallelOptions = new ParallelOptions();
                    var data = Tuple.Create<double[][], double[][], double[][], double[][]>(new double[hPrevious.Size][], new double[hPrevious.Size][], new double[hPrevious.Size][], new double[hPrevious.Size][]);
                    var length = this.hiddens * 3;

                    for (int i = 0; i < hPrevious.Size; i++)
                    {
                        data.Item4[i] = hPrevious[i];
                    }

                    parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                    Parallel.ForEach<double[], List<Tuple<long, double[], double[], double[], double[]>>>(x, parallelOptions, () => new List<Tuple<long, double[], double[], double[], double[]>>(), (vector, state, index, local) =>
                    {
                        var z = new double[this.hiddens];
                        var r = new double[this.hiddens];
                        var hHat = new double[this.hiddens];
                        var hNext = new double[this.hiddens];

                        for (int i = 0; i < this.hiddens; i++)
                        {
                            double sum1 = 0.0;
                            double sum2 = 0.0;

                            for (int j = 0; j < this.inputs; j++)
                            {
                                sum1 += vector[j] * this.xWeights[length * j + i];
                            }

                            for (int j = 0; j < this.hiddens; j++)
                            {
                                sum2 += hPrevious[index][j] * this.hWeights[length * j + i];
                            }

                            z[i] = this.sigmoidActivationFunction.Function(sum1 + sum2 + this.biases[i]);
                        }

                        for (int i = 0, j = this.hiddens; i < this.hiddens; i++, j++)
                        {
                            double sum1 = 0.0;
                            double sum2 = 0.0;

                            for (int k = 0; k < this.inputs; k++)
                            {
                                sum1 += vector[k] * this.xWeights[length * k + j];
                            }

                            for (int k = 0; k < this.hiddens; k++)
                            {
                                sum2 += hPrevious[index][k] * this.hWeights[length * k + j];
                            }

                            r[i] = this.sigmoidActivationFunction.Function(sum1 + sum2 + this.biases[j]);
                        }

                        for (int i = 0, j = this.hiddens * 2; i < this.hiddens; i++, j++)
                        {
                            double sum1 = 0.0;
                            double sum2 = 0.0;

                            for (int k = 0; k < this.inputs; k++)
                            {
                                sum1 += vector[k] * this.xWeights[length * k + j];
                            }

                            for (int k = 0; k < this.hiddens; k++)
                            {
                                sum2 += hPrevious[index][k] * r[i] * this.hWeights[length * k + j];
                            }

                            hHat[i] = this.tanhActivationFunction.Function(sum1 + sum2 + this.biases[j]);
                            hNext[i] = (1.0 - z[i]) * hPrevious[index][i] + z[i] * hHat[i];
                        }

                        local.Add(Tuple.Create<long, double[], double[], double[], double[]>(index, z, r, hHat, hNext));

                        return local;
                    }, (local) =>
                    {
                        lock (data)
                        {
                            local.ForEach(tuple =>
                            {
                                data.Item1[tuple.Item1] = tuple.Item2;
                                data.Item2[tuple.Item1] = tuple.Item3;
                                data.Item3[tuple.Item1] = tuple.Item4;
                                data.Item4[tuple.Item1] = tuple.Item5;
                            });
                        }
                    });

                    this.cache = Tuple.Create<Batch<double[]>, Batch<double[]>, double[][], double[][], double[][]>(x, hPrevious, data.Item1, data.Item2, data.Item3);

                    return new Batch<double[]>(data.Item4);
                }

                public Tuple<Batch<double[]>, Batch<double[]>, Batch<double[]>> Backward(Batch<double[]> dhNext)
                {
                    var parallelOptions = new ParallelOptions();
                    var length = this.hiddens * 3;
                    var x = this.cache.Item1;
                    var hPrevious = this.cache.Item2;
                    var z = this.cache.Item3;
                    var r = this.cache.Item4;
                    var hHat = this.cache.Item5;
                    var data = Tuple.Create<double[][], double[][], double[][], double[][], double[][]>(new double[dhNext.Size][], new double[dhNext.Size][], new double[dhNext.Size][], new double[dhNext.Size][], new double[dhNext.Size][]);
                    var vectorList = new List<double[]>();

                    parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                    Parallel.ForEach<double[], List<Tuple<long, double[], double[], double[], double[], double[]>>>(dhNext, parallelOptions, () => new List<Tuple<long, double[], double[], double[], double[], double[]>>(), (vector, state, index, local) =>
                    {
                        var dhHat = new double[this.hiddens];
                        var dhPrev = new double[this.hiddens];
                        var dt = new double[length];
                        var dhr = new double[this.hiddens];
                        var dWh = new double[this.hiddens * length];
                        var dWx = new double[this.inputs * length];
                        var dx = new double[this.inputs];

                        for (int i = 0, j = this.hiddens; i < this.hiddens; i++, j++)
                        {
                            dhHat[i] = vector[i] * z[index][i];
                            dhPrev[i] = vector[i] * (1.0 - z[index][i]);
                            
                            dt[i] = dhHat[i] * this.tanhActivationFunction.Derivative(hHat[index][i]);
                            dt[j] = (vector[i] * hHat[index][i] - vector[i] * hPrevious[index][i]) * this.sigmoidActivationFunction.Derivative(z[index][i]);
                        }

                        for (int i = 0, j = 0; i < this.hiddens; i++)
                        {
                            double error1 = 0.0;
                            double error2 = 0.0;

                            for (int k = 0, l = this.hiddens; k < this.hiddens; k++, l++)
                            {
                                var m = this.hiddens + j;

                                error1 += dt[k] * this.hWeights[j];
                                dWh[j] = dt[k] * r[index][i] * hPrevious[index][i];

                                error2 += dt[l] * this.hWeights[m];
                                dWh[m] = dt[l] * hPrevious[index][i];

                                j++;
                            }

                            dhr[i] = error1;
                            dhPrev[i] += r[index][i] * error1 + error2;
                        }

                        for (int i = 0, j = this.hiddens * 2; i < this.hiddens; i++, j++)
                        {
                            dt[j] = dhr[i] * hPrevious[index][i] * this.sigmoidActivationFunction.Derivative(r[index][i]);
                        }

                        for (int i = 0, j = 0, offset = this.hiddens * 2; i < this.hiddens; i++)
                        {
                            double error = 0.0;

                            for (int k = 0, l = offset; k < this.hiddens; k++, l++)
                            {
                                var m = offset + j;

                                error += dt[l] * this.hWeights[m];
                                dWh[m] = dt[l] * hPrevious[index][i];
                                j++;
                            }

                            dhPrev[i] += error;
                        }

                        for (int i = 0, j = 0; i < this.inputs; i++)
                        {
                            double error = 0.0;

                            for (int k = 0; k < length; k++)
                            {
                                error += dt[k] * this.xWeights[j];
                                dWx[j] = dt[k] * x[index][i];
                                j++;
                            }

                            dx[i] = error;
                        }

                        local.Add(Tuple.Create<long, double[], double[], double[], double[], double[]>(index, dt, dhPrev, dWh, dx, dWx));

                        return local;
                    }, (local) =>
                    {
                        lock (data)
                        {
                            local.ForEach(tuple =>
                            {
                                data.Item1[tuple.Item1] = tuple.Item2;
                                data.Item2[tuple.Item1] = tuple.Item3;
                                data.Item3[tuple.Item1] = tuple.Item4;
                                data.Item4[tuple.Item1] = tuple.Item5;
                                data.Item5[tuple.Item1] = tuple.Item6;
                            });
                        }
                    });

                    for (int j = 0; j < dhNext.Size; j++)
                    {
                        vectorList.Add(data.Item5[j].Concat<double>(data.Item3[j]).Concat<double>(data.Item1[j]).ToArray<double>());
                    }

                    return Tuple.Create<Batch<double[]>, Batch<double[]>, Batch<double[]>>(new Batch<double[]>(data.Item4), new Batch<double[]>(data.Item2), new Batch<double[]>(vectorList));
                }
            }
        }
    }
}
