using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Threading.Tasks;
using Megalopolis.ActivationFunctions;

namespace Megalopolis
{
    namespace Layers
    {
        // Long short-term memory (LSTM)
        [DataContract]
        public class LSTM : Layer, IUpdatable
        {
            [DataMember]
            private double[] weights = null;
            [DataMember]
            private double[] biases = null;
            [DataMember]
            private int timesteps = 0;
            [DataMember]
            private bool stateful = false;
            private ValueTuple<Batch<double[]>, Batch<double[]>> state = new ValueTuple<Batch<double[]>, Batch<double[]>>(null, null);
            private List<InternalLSTM> layerList = null;
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

            public ValueTuple<Batch<double[]>, Batch<double[]>> State
            {
                get
                {
                    return this.state;
                }
            }

            public LSTM(int inputs, int outputs, Func<int, int, double> func) : base(inputs, outputs)
            {
                var length1 = outputs * 4;
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

            public LSTM(Layer layer, int nodes, Func<int, int, double> func) : base(layer, nodes)
            {
                var length1 = outputs * 4;
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

            public LSTM(int nodes, Func<int, int, double> func, Layer layer) : base(nodes, layer)
            {
                var length1 = outputs * 4;
                var length2 = nodes * length1 + layer.Inputs * length1;

                this.weights = new double[length2];
                this.biases = new double[length1];
                this.tanhActivationFunction = new HyperbolicTangent();
                this.sigmoidActivationFunction = new Sigmoid();

                for (int i = 0; i < length2; i++)
                {
                    this.weights[i] = func(layer.Inputs, nodes);
                }

                for (int i = 0; i < length1; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                var length1 = this.outputs * 4;
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

                this.layerList = new List<InternalLSTM>();

                if (this.stateful)
                {
                    if (this.state.Item1 == null && this.state.Item2 == null)
                    {
                        this.state.Item1 = new Batch<double[]>(new double[inputs.Size][]);
                        this.state.Item2 = new Batch<double[]>(new double[inputs.Size][]);

                        for (int i = 0; i < inputs.Size; i++)
                        {
                            this.state.Item1[i] = new double[this.outputs];
                            this.state.Item2[i] = new double[this.outputs];

                            for (int j = 0; j < this.outputs; j++)
                            {
                                this.state.Item1[i][j] = 0.0;
                                this.state.Item2[i][j] = 0.0;
                            }
                        }
                    }
                    else if (this.state.Item1 == null)
                    {
                        this.state.Item1 = new Batch<double[]>(new double[inputs.Size][]);

                        for (int i = 0; i < inputs.Size; i++)
                        {
                            this.state.Item1[i] = new double[this.outputs];

                            for (int j = 0; j < this.outputs; j++)
                            {
                                this.state.Item1[i][j] = 0.0;
                            }
                        }
                    }
                    else if (this.state.Item2 == null)
                    {
                        this.state.Item2 = new Batch<double[]>(new double[inputs.Size][]);

                        for (int i = 0; i < inputs.Size; i++)
                        {
                            this.state.Item2[i] = new double[this.outputs];

                            for (int j = 0; j < this.outputs; j++)
                            {
                                this.state.Item2[i][j] = 0.0;
                            }
                        }
                    }
                }
                else
                {
                    this.state.Item1 = new Batch<double[]>(new double[inputs.Size][]);
                    this.state.Item2 = new Batch<double[]>(new double[inputs.Size][]);

                    for (int i = 0; i < inputs.Size; i++)
                    {
                        this.state.Item1[i] = new double[this.outputs];
                        this.state.Item2[i] = new double[this.outputs];

                        for (int j = 0; j < this.outputs; j++)
                        {
                            this.state.Item1[i][j] = 0.0;
                            this.state.Item2[i][j] = 0.0;
                        }
                    }
                }

                for (int t = 0; t < this.timesteps; t++)
                {
                    var layer = new InternalLSTM(this.inputs, this.outputs, xWeights, hWeights, this.biases, this.tanhActivationFunction, this.sigmoidActivationFunction);
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

                    var tuple = layer.Forward(x, this.state.Item1, this.state.Item2);

                    this.state.Item1 = tuple.Item1;
                    this.state.Item2 = tuple.Item2;

                    for (int i = 0; i < inputs.Size; i++)
                    {
                        for (int j = 0, k = this.outputs * t; j < this.outputs; j++, k++)
                        {
                            outputs[i][k] = this.state.Item1[i][j];
                        }
                    }

                    this.layerList.Add(layer);
                }

                return new Batch<double[]>(outputs);
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                var length1 = this.outputs * 4;
                var length2 = this.timesteps * this.inputs;
                var length3 = this.inputs * length1 + this.outputs * length1 + length1;
                var d = new double[deltas.Size][];
                var dh = new Batch<double[]>(new double[deltas.Size][]);
                var dc = new Batch<double[]>(new double[deltas.Size][]);

                this.gradients = new double[deltas.Size][];

                for (int i = 0; i < deltas.Size; i++)
                {
                    d[i] = new double[length2];
                    dh[i] = new double[this.outputs];
                    dc[i] = new double[this.outputs];

                    for (int j = 0; j < this.outputs; j++)
                    {
                        dh[i][j] = 0.0;
                        dc[i][j] = 0.0;
                    }

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

                    var tuple = this.layerList[t].Backward(dh, dc);

                    dh = tuple.Item2;
                    dc = tuple.Item3;

                    for (int i = 0; i < deltas.Size; i++)
                    {
                        for (int j = 0, k = this.inputs * t; j < this.inputs; j++, k++)
                        {
                            d[i][k] = tuple.Item1[i][j];
                        }

                        for (int j = 0; j < length3; j++)
                        {
                            this.gradients[i][j] += tuple.Item4[i][j];
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
                var length = this.outputs * 4;
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
                var length1 = this.outputs * 4;
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
                    this.weights[j] = func(this.weights[j], gradients[length2][j] / gradients.Size);
                }

                for (int i = 0, j = offset; i < length1; i++, j++)
                {
                    this.biases[i] = func(this.biases[i], gradients[0][j] / gradients.Size);
                }
            }

            private class InternalLSTM
            {
                private int inputs = 0;
                private int hiddens = 0;
                private double[] xWeights = null;
                private double[] hWeights = null;
                private double[] biases = null;
                private IActivationFunction tanhActivationFunction = null;
                private IActivationFunction sigmoidActivationFunction = null;
                private Tuple<Batch<double[]>, Batch<double[]>, Batch<double[]>, Tuple<double[][], double[][], double[][], double[][]>, double[][]> cache = null;

                public InternalLSTM(int inputs, int hiddens, double[] xWeights, double[] hWeights, double[] biases, IActivationFunction tanhActivationFunction, IActivationFunction sigmoidActivationFunction)
                {
                    this.inputs = inputs;
                    this.hiddens = hiddens;
                    this.xWeights = xWeights;
                    this.hWeights = hWeights;
                    this.biases = biases;
                    this.tanhActivationFunction = tanhActivationFunction;
                    this.sigmoidActivationFunction = sigmoidActivationFunction;
                }

                public Tuple<Batch<double[]>, Batch<double[]>> Forward(Batch<double[]> x, Batch<double[]> hPrevious, Batch<double[]> cPrevious)
                {
                    var parallelOptions = new ParallelOptions();
                    var data = Tuple.Create<double[][], double[][], double[][], double[][], double[][], double[][]>(new double[hPrevious.Size][], new double[hPrevious.Size][], new double[hPrevious.Size][], new double[hPrevious.Size][], new double[hPrevious.Size][], new double[hPrevious.Size][]);
                    var length = this.hiddens * 4;

                    parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                    Parallel.ForEach<double[], List<Tuple<long, double[], double[], double[], double[], double[], double[]>>>(hPrevious, parallelOptions, () => new List<Tuple<long, double[], double[], double[], double[], double[], double[]>>(), (vector, state, index, local) =>
                    {
                        var v = new double[length];
                        var f = new double[this.hiddens];
                        var g = new double[this.hiddens];
                        var i = new double[this.hiddens];
                        var o = new double[this.hiddens];
                        var cNext = new double[this.hiddens];
                        var hNext = new double[this.hiddens];

                        for (int j = 0; j < length; j++)
                        {
                            double sum = 0.0;

                            for (int k = 0; k < this.hiddens; k++)
                            {
                                sum += vector[k] * this.hWeights[length * k + j];
                            }

                            v[j] = sum;
                        }

                        for (int j = 0; j < this.hiddens; j++)
                        {
                            double sum = 0.0;

                            for (int k = 0; k < this.inputs; k++)
                            {
                                sum += x[index][k] * this.xWeights[length * k + j];
                            }

                            f[j] = this.sigmoidActivationFunction.Function(sum + v[j] + this.biases[j]);
                        }

                        for (int j = 0, k = this.hiddens; j < this.hiddens; j++)
                        {
                            double sum = 0.0;

                            for (int l = 0; l < this.inputs; l++)
                            {
                                sum += x[index][l] * this.xWeights[length * l + k];
                                k++;
                            }

                            g[j] = this.tanhActivationFunction.Function(sum + v[k] + this.biases[k]);
                        }

                        for (int j = 0, k = this.hiddens * 2; j < this.hiddens; j++)
                        {
                            double sum = 0.0;

                            for (int l = 0; l < this.inputs; l++)
                            {
                                sum += x[index][l] * this.xWeights[length * l + k];
                                k++;
                            }

                            i[j] = this.sigmoidActivationFunction.Function(sum + v[k] + this.biases[k]);
                        }

                        for (int j = 0, k = this.hiddens * 3; j < this.hiddens; j++)
                        {
                            double sum = 0.0;

                            for (int l = 0; l < this.inputs; l++)
                            {
                                sum += x[index][l] * this.xWeights[length * l + k];
                                k++;
                            }

                            o[j] = this.sigmoidActivationFunction.Function(sum + v[k] + this.biases[k]);
                        }

                        for (int j = 0; j < this.hiddens; j++)
                        {
                            cNext[j] = f[j] * cPrevious[index][j] + g[j] + i[j];
                            hNext[j] = o[j] * this.tanhActivationFunction.Function(cNext[j]);
                        }

                        local.Add(Tuple.Create<long, double[], double[], double[], double[], double[], double[]>(index, f, g, i, o, hNext, cNext));

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
                                data.Item6[tuple.Item1] = tuple.Item7;
                            });
                        }
                    });

                    this.cache = Tuple.Create<Batch<double[]>, Batch<double[]>, Batch<double[]>, Tuple<double[][], double[][], double[][], double[][]>, double[][]>(x, hPrevious, cPrevious, Tuple.Create<double[][], double[][], double[][], double[][]>(data.Item1, data.Item2, data.Item3, data.Item4), data.Item6);

                    return Tuple.Create<Batch<double[]>, Batch<double[]>>(new Batch<double[]>(data.Item5), new Batch<double[]>(data.Item6));
                }

                public Tuple<Batch<double[]>, Batch<double[]>, Batch<double[]>, Batch<double[]>> Backward(Batch<double[]> dhNext, Batch<double[]> dcNext)
                {
                    var parallelOptions = new ParallelOptions();
                    var length = this.hiddens * 4;
                    var x = this.cache.Item1;
                    var hPrevious = this.cache.Item2;
                    var cPrevious = this.cache.Item3;
                    var f = this.cache.Item4.Item1;
                    var g = this.cache.Item4.Item2;
                    var i = this.cache.Item4.Item3;
                    var o = this.cache.Item4.Item4;
                    var cNext = this.cache.Item5;
                    var data = Tuple.Create<double[][], double[][], double[][], double[][], double[][], double[][]>(new double[dhNext.Size][], new double[dhNext.Size][], new double[dhNext.Size][], new double[dhNext.Size][], new double[dhNext.Size][], new double[dhNext.Size][]);
                    var vectorList = new List<double[]>();

                    parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                    Parallel.ForEach<double[], List<Tuple<long, double[], double[], double[], double[], double[], double[]>>>(dhNext, parallelOptions, () => new List<Tuple<long, double[], double[], double[], double[], double[], double[]>>(), (vector, state, index, local) =>
                    {
                        var dcPrevious = new double[this.hiddens];
                        var dA = new double[length];
                        var dWh = new double[this.hiddens * length];
                        var dWx = new double[this.inputs * this.hiddens];
                        var dhPrev = new double[this.hiddens];
                        var dx = new double[this.inputs];

                        for (int j = 0, k = this.hiddens, l = this.hiddens * 2, m = this.hiddens * 3; j < this.hiddens; j++, k++, l++, m++)
                        {
                            var tanh = this.tanhActivationFunction.Function(cNext[index][j]);
                            var ds = dcNext[index][j] + vector[j] * o[index][j] * this.tanhActivationFunction.Derivative(tanh);

                            dcPrevious[j] = ds * f[index][j];
                            dA[j] = ds * cPrevious[index][j] * this.sigmoidActivationFunction.Derivative(i[index][j]); // df
                            dA[k] = ds * i[index][j] * this.tanhActivationFunction.Derivative(g[index][j]); // dg
                            dA[l] = ds * g[index][j] * this.sigmoidActivationFunction.Derivative(i[index][j]); // di
                            dA[m] = vector[j] * tanh * this.sigmoidActivationFunction.Derivative(o[index][j]); // do
                        }

                        for (int j = 0, k = 0; j < this.hiddens; j++)
                        {
                            double error = 0.0;

                            for (int l = 0; l < length; l++)
                            {
                                error += dA[l] * this.hWeights[k];
                                dWh[k] = dA[l] * hPrevious[index][j];
                                k++;
                            }

                            dhPrev[j] = error;
                        }

                        for (int j = 0, k = 0; j < this.inputs; j++)
                        {
                            double error = 0.0;

                            for (int l = 0; l < length; l++)
                            {
                                error += dA[l] * this.xWeights[k];
                                dWx[k] = dA[l] * x[index][j];
                                k++;
                            }

                            dx[j] = error;
                        }

                        local.Add(Tuple.Create<long, double[], double[], double[], double[], double[], double[]>(index, dA, dhPrev, dcPrevious, dWh, dx, dWx));

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
                                data.Item6[tuple.Item1] = tuple.Item7;
                            });
                        }
                    });

                    for (int j = 0; j < dhNext.Size; j++)
                    {
                        vectorList.Add(data.Item6[j].Concat<double>(data.Item4[j]).Concat<double>(data.Item1[j]).ToArray<double>());
                    }

                    return Tuple.Create<Batch<double[]>, Batch<double[]>, Batch<double[]>, Batch<double[]>>(new Batch<double[]>(data.Item5), new Batch<double[]>(data.Item2), new Batch<double[]>(data.Item3), new Batch<double[]>(vectorList));
                }
            }
        }
    }
}
