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
        [DataContract]
        public class Recurrent : Layer, IUpdatable
        {
            [DataMember]
            private double[] weights = null;
            [DataMember]
            private double[] biases = null;
            [DataMember]
            private int timesteps = 0;
            [DataMember]
            private bool stateful = false;
            private List<InternalRecurrent> layerList = null;
            private Batch<double[]> h = null;
            private Batch<double[]> dh = null;
            private double[][] gradients = null;
            [DataMember]
            private IActivationFunction tanhActivationFunction = null;

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

            public Recurrent(int inputs, int outputs, int timesteps, bool stateful, Func<int, int, double> func) : base(inputs, outputs)
            {
                var length = inputs * outputs + outputs * outputs;

                this.weights = new double[length];
                this.biases = new double[outputs];
                this.timesteps = timesteps;
                this.stateful = stateful;
                this.tanhActivationFunction = new HyperbolicTangent();

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(inputs, outputs);
                }

                for (int i = 0; i < outputs; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public Recurrent(Layer layer, int nodes, int timesteps, bool stateful, Func<int, int, double> func) : base(layer, nodes)
            {
                var length = layer.Outputs * nodes + nodes * nodes;

                this.weights = new double[length];
                this.biases = new double[nodes];
                this.timesteps = timesteps;
                this.stateful = stateful;
                this.tanhActivationFunction = new HyperbolicTangent();

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(layer.Outputs, nodes);
                }

                for (int i = 0; i < nodes; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public Recurrent(int nodes, int timesteps, bool stateful, Func<int, int, double> func, Layer layer) : base(nodes, layer)
            {
                var length = nodes * nodes + layer.Inputs * nodes;

                this.weights = new double[length];
                this.biases = new double[nodes];
                this.timesteps = timesteps;
                this.stateful = stateful;
                this.tanhActivationFunction = new HyperbolicTangent();

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(layer.Inputs, nodes);
                }

                for (int i = 0; i < nodes; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                var length1 = this.inputs * this.outputs;
                var length2 = this.outputs * this.outputs;
                var length3 = this.timesteps * this.outputs;
                var xWeights = new double[length1];
                var hWeights = new double[length2];
                var outputs = new double[inputs.Size][];

                for (int i = 0; i < length1; i++)
                {
                    xWeights[i] = this.weights[i];
                }

                for (int i = 0, j = length1; i < length2; i++, j++)
                {
                    hWeights[i] = this.weights[j];
                }

                for (int i = 0; i < inputs.Size; i++)
                {
                    outputs[i] = new double[length3];
                }

                this.layerList = new List<InternalRecurrent>();

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
                    var layer = new InternalRecurrent(this.inputs, this.outputs, xWeights, hWeights, this.biases, this.tanhActivationFunction);
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
                // Truncated Backpropagation Through Time (Truncated BPTT)
                var length1 = this.timesteps * this.inputs;
                var length2 = this.inputs * this.outputs + this.outputs * this.outputs + this.outputs;
                var d = new double[deltas.Size][];
                var dh = new Batch<double[]>(new double[deltas.Size][]);

                this.gradients = new double[deltas.Size][];

                for (int i = 0; i < deltas.Size; i++)
                {
                    d[i] = new double[length1];
                    dh[i] = new double[this.outputs];

                    for (int j = 0; j < this.outputs; j++)
                    {
                        dh[i][j] = 0.0;
                    }

                    for (int j = 0; j < length2; j++)
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
                var length = this.inputs * this.outputs * this.outputs * this.outputs;

                foreach (double[] vector in this.gradients)
                {
                    for (int i = 0; i < length; i++)
                    {
                        vector[i] = func(true, vector[i], i);
                    }

                    for (int i = 0, j = length; i < this.outputs; i++, j++)
                    {
                        vector[j] = func(false, vector[j], i);
                    }
                }
            }

            public void Update(Batch<double[]> gradients, Func<double, double, double> func)
            {
                var length1 = this.inputs * this.outputs;
                var length2 = this.outputs * this.outputs;
                var offset = length1 + length2;

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

                    for (int j = 0, k = offset; j < this.outputs; j++, k++)
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
                    this.weights[j] = func(this.weights[j], gradients[length1][j] / gradients.Size);
                }

                for (int i = 0, j = offset; i < this.outputs; i++, j++)
                {
                    this.biases[i] = func(this.biases[i], gradients[0][j] / gradients.Size);
                }
            }

            private class InternalRecurrent
            {
                private int inputs = 0;
                private int hiddens = 0;
                private double[] xWeights = null;
                private double[] hWeights = null;
                private double[] biases = null;
                private IActivationFunction activationFunction = null;
                private Tuple<Batch<double[]>, Batch<double[]>, double[][]> cache = null;

                public InternalRecurrent(int inputs, int hiddens, double[] xWeights, double[] hWeights, double[] biases, IActivationFunction activationFunction)
                {
                    this.inputs = inputs;
                    this.hiddens = hiddens;
                    this.xWeights = xWeights;
                    this.hWeights = hWeights;
                    this.biases = biases;
                    this.activationFunction = activationFunction;
                }

                public Batch<double[]> Forward(Batch<double[]> x, Batch<double[]> hPrevious)
                {
                    // h(t) = tanh(h(t-1) Wh + x(t) Wx + b)
                    var parallelOptions = new ParallelOptions();
                    var data = new double[hPrevious.Size][];

                    parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                    Parallel.ForEach<double[], List<Tuple<long, double[]>>>(hPrevious, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector, state, index, local) =>
                    {
                        var v = new double[this.hiddens];
                        var hNext = new double[this.hiddens];

                        for (int i = 0; i < this.hiddens; i++)
                        {
                            double sum = 0.0;

                            for (int j = 0; j < this.hiddens; j++)
                            {
                                sum += vector[j] * this.hWeights[this.hiddens * j + i];
                            }

                            v[i] = sum;
                        }

                        for (int i = 0; i < this.hiddens; i++)
                        {
                            double sum = 0.0;

                            for (int j = 0; j < this.inputs; j++)
                            {
                                sum += x[index][j] * this.xWeights[this.hiddens * j + i];
                            }

                            hNext[i] = this.activationFunction.Function(sum + v[i] + this.biases[i]);
                        }

                        local.Add(Tuple.Create<long, double[]>(index, hNext));

                        return local;
                    }, (local) =>
                    {
                        lock (data)
                        {
                            local.ForEach(tuple =>
                            {
                                data[tuple.Item1] = tuple.Item2;
                            });
                        }
                    });

                    this.cache = Tuple.Create<Batch<double[]>, Batch<double[]>, double[][]>(x, hPrevious, data);

                    return new Batch<double[]>(data);
                }

                public Tuple<Batch<double[]>, Batch<double[]>, Batch<double[]>> Backward(Batch<double[]> dhNext)
                {
                    var parallelOptions = new ParallelOptions();
                    var x = this.cache.Item1;
                    var hPrevious = this.cache.Item2;
                    var hNext = this.cache.Item3;
                    var data = Tuple.Create<double[][], double[][], double[][], double[][], double[][]>(new double[dhNext.Size][], new double[dhNext.Size][], new double[dhNext.Size][], new double[dhNext.Size][], new double[dhNext.Size][]);
                    var vectorList = new List<double[]>();

                    parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                    Parallel.ForEach<double[], List<Tuple<long, double[], double[], double[], double[], double[]>>>(dhNext, parallelOptions, () => new List<Tuple<long, double[], double[], double[], double[], double[]>>(), (vector, state, index, local) =>
                    {
                        var dt = new double[this.hiddens];
                        var dWh = new double[this.hiddens * this.hiddens];
                        var dWx = new double[this.inputs * this.hiddens];
                        var dhPrev = new double[this.hiddens];
                        var dx = new double[this.inputs];
                        
                        for (int i = 0; i < this.hiddens; i++)
                        {
                            dt[i] = this.activationFunction.Derivative(hNext[index][i]) * vector[i];
                        }

                        for (int i = 0, j = 0; i < this.hiddens; i++)
                        {
                            double error = 0.0;

                            for (int k = 0; k < this.hiddens; k++)
                            {
                                error += dt[k] * this.hWeights[j];
                                dWh[j] = dt[k] * hPrevious[index][i];
                                j++;
                            }

                            dhPrev[i] = error;
                        }

                        for (int i = 0, j = 0; i < this.inputs; i++)
                        {
                            double error = 0.0;

                            for (int k = 0; k < this.hiddens; k++)
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

                    for (int i = 0; i < dhNext.Size; i++)
                    {
                        vectorList.Add(data.Item5[i].Concat<double>(data.Item3[i]).Concat<double>(data.Item1[i]).ToArray<double>());
                    }

                    return Tuple.Create<Batch<double[]>, Batch<double[]>, Batch<double[]>>(new Batch<double[]>(data.Item4), new Batch<double[]>(data.Item2), new Batch<double[]>(vectorList));
                }
            }
        }
    }
}
