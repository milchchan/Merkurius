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
            private RecurrentCore forwardRecurrent = null;
            [DataMember]
            private RecurrentCore backwardRecurrent = null;
            [DataMember]
            private double[] weights = null;
            [DataMember]
            private double[] biases = null;
            private Batch<double[]> state = null;
            private Batch<double[]> deltaState = null;

            public double[] Weights
            {
                get
                {
                    if (this.backwardRecurrent == null)
                    {
                        return this.forwardRecurrent.Weights;
                    }

                    return this.weights;
                }
                set
                {
                    if (this.backwardRecurrent == null)
                    {
                        this.forwardRecurrent.Weights = value;
                    }
                    else
                    {
                        this.weights = value;
                    }
                }
            }

            public double[] Biases
            {
                get
                {
                    if (this.backwardRecurrent == null)
                    {
                        return this.forwardRecurrent.Biases;
                    }

                    return this.biases;
                }
                set
                {
                    if (this.backwardRecurrent == null)
                    {
                        this.forwardRecurrent.Biases = value;
                    }
                    else
                    {
                        this.biases = value;
                    }
                }
            }

            public int Timesteps
            {
                get
                {
                    return this.forwardRecurrent.Timesteps;
                }
            }

            public Batch<double[]> State
            {
                get
                {
                    if (this.backwardRecurrent == null)
                    {
                        return this.forwardRecurrent.State;
                    }

                    return this.state;
                }
                set
                {
                    if (this.backwardRecurrent == null)
                    {
                        this.forwardRecurrent.State = value;
                    }

                    this.state = value;
                }
            }

            public Batch<double[]> DeltaState
            {
                get
                {
                    return this.deltaState;
                }
            }

            public Recurrent(int inputs, int hiddens, int timesteps, bool stateful, bool bidirectional, Func<int, int, double> func) : base(inputs, hiddens)
            {
                var length1 = hiddens * 3;
                var length2 = inputs * length1 + hiddens * length1;

                this.forwardRecurrent = new RecurrentCore(inputs, hiddens, timesteps, stateful, func);

                if (bidirectional)
                {
                    var length3 = length1 * 2;

                    this.backwardRecurrent = new RecurrentCore(inputs, hiddens, timesteps, stateful, func);
                    this.weights = new double[length2 * 2];
                    this.biases = new double[length3];

                    for (int i = 0, j = length2; i < length2; i++, j++)
                    {
                        this.weights[i] = this.forwardRecurrent.Weights[i];
                        this.weights[j] = this.backwardRecurrent.Weights[i];
                    }

                    for (int i = 0; i < length3; i++)
                    {
                        this.biases[i] = 0.0;
                    }
                }
            }

            public Recurrent(int inputs, int hiddens, int timesteps, bool stateful, bool bidirectional, Func<int, int, double> func, Layer layer) : base(inputs, layer)
            {
                var length1 = hiddens * 3;
                var length2 = inputs * length1 + hiddens * length1;

                this.forwardRecurrent = new RecurrentCore(inputs, hiddens, timesteps, stateful, func);

                if (bidirectional)
                {
                    var length3 = length1 * 2;

                    this.backwardRecurrent = new RecurrentCore(inputs, hiddens, timesteps, stateful, func);
                    this.weights = new double[length2 * 2];
                    this.biases = new double[length3];

                    for (int i = 0, j = length2; i < length2; i++, j++)
                    {
                        this.weights[i] = this.forwardRecurrent.Weights[i];
                        this.weights[j] = this.backwardRecurrent.Weights[i];
                    }

                    for (int i = 0; i < length3; i++)
                    {
                        this.biases[i] = 0.0;
                    }
                }
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                if (this.backwardRecurrent == null)
                {
                    return this.forwardRecurrent.Forward(inputs, isTraining);
                }

                for (int i = 0, length = this.weights.Length / 2; i < length; i++)
                {
                    this.forwardRecurrent.Weights[i] = this.weights[i];
                    this.backwardRecurrent.Weights[i] = this.weights[i + length];
                }

                for (int i = 0, length = this.biases.Length / 2; i < length; i++)
                {
                    this.forwardRecurrent.Biases[i] = this.biases[i];
                    this.backwardRecurrent.Biases[i] = this.biases[i + length];
                }

                if (this.state != null)
                {
                    for (int i = 0; i < this.state.Size; i++)
                    {
                        int length = this.state[i].Length / 2;

                        for (int j = 0; j < length; j++)
                        {
                            this.forwardRecurrent.State[i][j] = this.state[i][j];
                            this.backwardRecurrent.State[i][j] = this.state[i][j + length];
                        }
                    }
                }

                var outputs1 = this.forwardRecurrent.Forward(inputs, isTraining);
                var outputs2 = this.backwardRecurrent.Forward(Reverse(inputs), isTraining);
                var vectorList1 = new List<double[]>();
                var vectorList2 = new List<double[]>();

                for (int i = 0; i < outputs1.Size; i++)
                {
                    var vector = new double[outputs1[i].Length];

                    for (int j = 0; j < outputs1[i].Length; j++)
                    {
                        vector[j] = outputs1[i][j] + outputs2[i][j];
                    }

                    vectorList1.Add(vector);
                }

                for (int i = 0; i < this.forwardRecurrent.State.Size; i++)
                {
                    int length = this.forwardRecurrent.State[i].Length;
                    var vector = new double[this.forwardRecurrent.State[i].Length + this.backwardRecurrent.State[i].Length];

                    for (int j = 0; j < length; j++)
                    {
                        vector[j] = this.forwardRecurrent.State[i][j];
                        vector[j + length] = this.backwardRecurrent.State[i][j];
                    }

                    vectorList2.Add(vector);
                }

                this.state = new Batch<double[]>(vectorList2);

                return new Batch<double[]>(vectorList1);
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                if (this.backwardRecurrent == null)
                {
                    return this.forwardRecurrent.Backward(deltas);
                }

                var dx1 = this.forwardRecurrent.Backward(deltas);
                var dx2 = this.backwardRecurrent.Backward(Reverse(deltas));
                var vectorList = new List<double[]>();

                for (int i = 0; i < dx1.Size; i++)
                {
                    var vector = new double[dx1[i].Length];

                    for (int j = 0; j < dx1[i].Length; j++)
                    {
                        vector[j] = dx1[i][j] + dx2[i][j];
                    }

                    vectorList.Add(vector);
                }

                this.deltaState = new Batch<double[]>(new double[this.forwardRecurrent.DeltaState.Size][]);

                for (int i = 0; i < this.forwardRecurrent.DeltaState.Size; i++)
                {
                    this.deltaState[i] = new double[forwardRecurrent.DeltaState[i].Length];

                    for (int j = 0; j < this.forwardRecurrent.DeltaState[i].Length; j++)
                    {
                        this.deltaState[i][j] = this.forwardRecurrent.DeltaState[i][j] + this.backwardRecurrent.DeltaState[i][j];
                    }
                }

                return new Batch<double[]>(vectorList);
            }

            public Batch<double[]> GetGradients()
            {
                if (this.backwardRecurrent == null)
                {
                    return this.forwardRecurrent.GetGradients();
                }

                var gradients1 = this.forwardRecurrent.GetGradients();
                var gradients2 = this.backwardRecurrent.GetGradients();
                var vectorList = new List<double[]>();

                for (int i = 0; i < gradients1.Size; i++)
                {
                    var vector = new double[gradients1[i].Length + gradients2[i].Length];

                    for (int j = 0; j < gradients1[i].Length; j++)
                    {
                        vector[j] = gradients1[i][j];
                        vector[j + gradients1[i].Length] = gradients2[i][j];
                    }

                    vectorList.Add(vector);
                }

                return new Batch<double[]>(vectorList);
            }

            public void SetGradients(Func<bool, double, int, double> func)
            {
                this.forwardRecurrent.SetGradients(func);

                if (this.backwardRecurrent != null)
                {
                    this.backwardRecurrent.SetGradients((x, y, z) => func(x, y, this.weights.Length / 2 - 1 - z));
                }
            }

            public void Update(Batch<double[]> gradients, Func<double, double, double> func)
            {
                if (this.backwardRecurrent == null)
                {
                    this.forwardRecurrent.Update(gradients, func);
                }
                else
                {
                    var vectorList1 = new List<double[]>();
                    var vectorList2 = new List<double[]>();

                    foreach (var vector in gradients)
                    {
                        int length = vector.Length / 2;
                        var vector1 = new double[length];
                        var vector2 = new double[length];

                        for (int i = 0, j = length; i < length; i++, j++)
                        {
                            vector1[i] = vector[i];
                            vector2[i] = vector[j];
                        }

                        vectorList1.Add(vector1);
                        vectorList2.Add(vector2);
                    }

                    this.forwardRecurrent.Update(new Batch<double[]>(vectorList1), func);
                    this.backwardRecurrent.Update(new Batch<double[]>(vectorList2), func);

                    for (int i = 0, length = this.weights.Length / 2; i < length; i++)
                    {
                        this.weights[i] = this.forwardRecurrent.Weights[i];
                        this.weights[i + length] = this.backwardRecurrent.Weights[i];
                    }

                    for (int i = 0, length = this.biases.Length / 2; i < length; i++)
                    {
                        this.biases[i] = this.forwardRecurrent.Biases[i];
                        this.biases[i + length] = this.backwardRecurrent.Biases[i];
                    }
                }
            }

            private Batch<double[]> Reverse(Batch<double[]> batch)
            {
                var vectorList = new List<double[]>();

                foreach (var vector1 in batch)
                {
                    var vector2 = new double[vector1.Length];

                    for (int i = 0, last = vector1.Length - 1; i < vector1.Length; i++)
                    {
                        vector2[i] = vector1[last - i];
                    }

                    vectorList.Add(vector2);
                }

                return new Batch<double[]>(vectorList);
            }

            [DataContract]
            public class RecurrentCore : Layer, IUpdatable
            {
                [DataMember]
                private double[] weights = null;
                [DataMember]
                private double[] biases = null;
                [DataMember]
                private int timesteps = 0;
                [DataMember]
                private bool stateful = false;
                private List<RecurrentCell> layerList = null;
                private Batch<double[]> h = null; // Hidden state
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

                public int Timesteps
                {
                    get
                    {
                        return this.timesteps;
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

                public Batch<double[]> DeltaState
                {
                    get
                    {
                        return this.dh;
                    }
                    set
                    {
                        this.dh = value;
                    }
                }

                public RecurrentCore(int inputs, int hiddens, int timesteps, bool stateful, Func<int, int, double> func) : base(inputs, hiddens)
                {
                    var length = inputs * hiddens + hiddens * hiddens;

                    this.weights = new double[length];
                    this.biases = new double[hiddens];
                    this.timesteps = timesteps;
                    this.stateful = stateful;
                    this.tanhActivationFunction = new HyperbolicTangent();

                    for (int i = 0; i < length; i++)
                    {
                        this.weights[i] = func(inputs, hiddens);
                    }

                    for (int i = 0; i < hiddens; i++)
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

                    this.layerList = new List<RecurrentCell>();

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
                    else if (this.h.Size < inputs.Size)
                    {
                        var batch = new Batch<double[]>(new double[inputs.Size][]);

                        for (int i = 0; i < this.h.Size; i++)
                        {
                            batch[i] = this.h[i];
                        }

                        for (int i = this.h.Size; i < inputs.Size; i++)
                        {
                            batch[i] = new double[this.outputs];

                            for (int j = 0; j < this.outputs; j++)
                            {
                                batch[i][j] = 0.0;
                            }
                        }

                        this.h = batch;
                    }

                    for (int t = 0; t < this.timesteps; t++)
                    {
                        var layer = new RecurrentCell(this.inputs, this.outputs, xWeights, hWeights, this.biases, this.tanhActivationFunction);
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

                        this.gradients[i] = new double[length2];

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
                    var length = this.inputs * this.outputs + this.outputs * this.outputs;

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
                        this.weights[j] = func(this.weights[j], gradients[0][j] / gradients.Size);
                    }

                    for (int i = 0, j = offset; i < this.outputs; i++, j++)
                    {
                        this.biases[i] = func(this.biases[i], gradients[0][j] / gradients.Size);
                    }
                }

                private class RecurrentCell
                {
                    private int inputs = 0;
                    private int hiddens = 0;
                    private double[] xWeights = null;
                    private double[] hWeights = null;
                    private double[] biases = null;
                    private IActivationFunction activationFunction = null;
                    private Tuple<Batch<double[]>, Batch<double[]>, double[][]> cache = null;

                    public RecurrentCell(int inputs, int hiddens, double[] xWeights, double[] hWeights, double[] biases, IActivationFunction activationFunction)
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

                        for (int i = 0; i < hPrevious.Size; i++)
                        {
                            data[i] = hPrevious[i];
                        }

                        parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                        Parallel.ForEach<double[], List<Tuple<long, double[]>>>(x, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector, state, index, local) =>
                        {
                            var v = new double[this.hiddens];
                            var hNext = new double[this.hiddens];

                            for (int i = 0; i < this.hiddens; i++)
                            {
                                double sum = 0.0;

                                for (int j = 0; j < this.hiddens; j++)
                                {
                                    sum += hPrevious[index][j] * this.hWeights[this.hiddens * j + i];
                                }

                                v[i] = sum;
                            }

                            for (int i = 0; i < this.hiddens; i++)
                            {
                                double sum = 0.0;

                                for (int j = 0; j < this.inputs; j++)
                                {
                                    sum += vector[j] * this.xWeights[this.hiddens * j + i];
                                }

                                hNext[i] = this.activationFunction.Forward(sum + v[i] + this.biases[i]);
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
                                dt[i] = this.activationFunction.Backward(hNext[index][i]) * vector[i];
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
}
