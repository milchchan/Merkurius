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
            private GRUCore? forwardGru = null;
            [DataMember]
            private GRUCore? backwardGru = null;
            [DataMember]
            private double[]? weights = null;
            [DataMember]
            private double[]? biases = null;
            private Batch<double[]>? state = null;
            private Batch<double[]>? deltaState = null;

            public double[] Weights
            {
                get
                {
                    if (this.backwardGru == null)
                    {
                        return this.forwardGru!.Weights;
                    }

                    return this.weights!;
                }
                set
                {
                    if (this.backwardGru == null)
                    {
                        this.forwardGru!.Weights = value;
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
                    if (this.backwardGru == null)
                    {
                        return this.forwardGru!.Biases;
                    }

                    return this.biases!;
                }
                set
                {
                    if (this.backwardGru == null)
                    {
                        this.forwardGru!.Biases = value;
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
                    return this.forwardGru!.Timesteps;
                }
            }

            public Batch<double[]>? State
            {
                get
                {
                    if (this.backwardGru == null)
                    {
                        return this.forwardGru!.State;
                    }

                    return this.state;
                }
                set
                {
                    if (this.backwardGru == null)
                    {
                        this.forwardGru!.State = value;
                    }

                    this.state = value;
                }
            }

            public Batch<double[]>? DeltaState
            {
                get
                {
                    return this.deltaState;
                }
            }

            public GRU(int inputs, int hiddens, int timesteps, bool stateful, bool bidirectional, Func<int, int, double> func) : base(inputs, hiddens)
            {
                var length1 = hiddens * 3;
                var length2 = inputs * length1 + hiddens * length1;

                this.forwardGru = new GRUCore(inputs, hiddens, timesteps, stateful, func);

                if (bidirectional)
                {
                    var length3 = length1 * 2;

                    this.backwardGru = new GRUCore(inputs, hiddens, timesteps, stateful, func);
                    this.weights = new double[length2 * 2];
                    this.biases = new double[length3];

                    for (int i = 0, j = length2; i < length2; i++, j++)
                    {
                        this.weights[i] = this.forwardGru.Weights[i];
                        this.weights[j] = this.backwardGru.Weights[i];
                    }

                    for (int i = 0; i < length3; i++)
                    {
                        this.biases[i] = 0.0;
                    }
                }
            }

            public GRU(int inputs, int hiddens, int timesteps, bool stateful, bool bidirectional, Func<int, int, double> func, Layer layer) : base(inputs, layer)
            {
                var length1 = hiddens * 3;
                var length2 = inputs * length1 + hiddens * length1;

                this.forwardGru = new GRUCore(inputs, hiddens, timesteps, stateful, func);

                if (bidirectional)
                {
                    var length3 = length1 * 2;

                    this.backwardGru = new GRUCore(inputs, hiddens, timesteps, stateful, func);
                    this.weights = new double[length2 * 2];
                    this.biases = new double[length3];

                    for (int i = 0, j = length2; i < length2; i++, j++)
                    {
                        this.weights[i] = this.forwardGru.Weights[i];
                        this.weights[j] = this.backwardGru.Weights[i];
                    }

                    for (int i = 0; i < length3; i++)
                    {
                        this.biases[i] = 0.0;
                    }
                }
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                if (this.backwardGru == null)
                {
                    return this.forwardGru!.Forward(inputs, isTraining);
                }

                for (int i = 0, length = this.weights!.Length / 2; i < length; i++)
                {
                    this.forwardGru!.Weights[i] = this.weights[i];
                    this.backwardGru.Weights[i] = this.weights[i + length];
                }

                for (int i = 0, length = this.biases!.Length / 2; i < length; i++)
                {
                    this.forwardGru!.Biases[i] = this.biases[i];
                    this.backwardGru.Biases[i] = this.biases[i + length];
                }

                if (this.state != null)
                {
                    for (int i = 0; i < this.state.Size; i++)
                    {
                        int length = this.state[i].Length / 2;

                        for (int j = 0; j < length; j++)
                        {
                            this.forwardGru!.State![i][j] = this.state[i][j];
                            this.backwardGru.State![i][j] = this.state[i][j + length];
                        }
                    }
                }

                var outputs1 = this.forwardGru!.Forward(inputs, isTraining);
                var outputs2 = this.backwardGru.Forward(Reverse(inputs), isTraining);
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

                for (int i = 0; i < this.forwardGru.State!.Size; i++)
                {
                    int length = this.forwardGru.State[i].Length;
                    var vector = new double[this.forwardGru.State[i].Length + this.backwardGru.State![i].Length];

                    for (int j = 0; j < length; j++)
                    {
                        vector[j] = this.forwardGru.State[i][j];
                        vector[j + length] = this.backwardGru.State[i][j];
                    }

                    vectorList2.Add(vector);
                }

                this.state = new Batch<double[]>(vectorList2);

                return new Batch<double[]>(vectorList1);
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                if (this.backwardGru == null)
                {
                    return this.forwardGru!.Backward(deltas);
                }

                var dx1 = this.forwardGru!.Backward(deltas);
                var dx2 = this.backwardGru.Backward(Reverse(deltas));
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

                this.deltaState = new Batch<double[]>(new double[this.forwardGru.DeltaState!.Size][]);

                for (int i = 0; i < this.forwardGru.DeltaState.Size; i++)
                {
                    this.deltaState[i] = new double[forwardGru.DeltaState[i].Length];

                    for (int j = 0; j < this.forwardGru.DeltaState[i].Length; j++)
                    {
                        this.deltaState[i][j] = this.forwardGru.DeltaState[i][j] + this.backwardGru.DeltaState![i][j];
                    }
                }

                return new Batch<double[]>(vectorList);
            }

            public Batch<double[]> GetGradients()
            {
                if (this.backwardGru == null)
                {
                    return this.forwardGru!.GetGradients();
                }

                var gradients1 = this.forwardGru!.GetGradients();
                var gradients2 = this.backwardGru.GetGradients();
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
                this.forwardGru!.SetGradients(func);

                if (this.backwardGru != null)
                {
                    this.backwardGru.SetGradients((x, y, z) => func(x, y, this.weights!.Length / 2 - 1 - z));
                }
            }

            public void Update(Batch<double[]> gradients, Func<double, double, double> func)
            {
                if (this.backwardGru == null)
                {
                    this.forwardGru!.Update(gradients, func);
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

                    this.forwardGru!.Update(new Batch<double[]>(vectorList1), func);
                    this.backwardGru.Update(new Batch<double[]>(vectorList2), func);

                    for (int i = 0, length = this.weights!.Length / 2; i < length; i++)
                    {
                        this.weights[i] = this.forwardGru.Weights[i];
                        this.weights[i + length] = this.backwardGru.Weights[i];
                    }

                    for (int i = 0, length = this.biases!.Length / 2; i < length; i++)
                    {
                        this.biases[i] = this.forwardGru.Biases[i];
                        this.biases[i + length] = this.backwardGru.Biases[i];
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
            private class GRUCore : Layer, IUpdatable
            {
                [DataMember]
                private double[]? weights = null;
                [DataMember]
                private double[]? biases = null;
                [DataMember]
                private int timesteps = 0;
                [DataMember]
                private bool stateful = false;
                private List<GRUCell>? layerList = null;
                private Batch<double[]>? h = null; // Hidden state
                private Batch<double[]>? dh = null;
                private double[][]? gradients = null;
                [DataMember]
                private IActivationFunction? tanhActivationFunction = null;
                [DataMember]
                private IActivationFunction? sigmoidActivationFunction = null;

                public double[] Weights
                {
                    get
                    {
                        return this.weights!;
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
                        return this.biases!;
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

                public Batch<double[]>? State
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

                public Batch<double[]>? DeltaState
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

                public GRUCore(int inputs, int hiddens, int timesteps, bool stateful, Func<int, int, double> func) : base(inputs, hiddens)
                {
                    var length1 = hiddens * 3;
                    var length2 = inputs * length1 + hiddens * length1;

                    this.weights = new double[length2];
                    this.biases = new double[length1];
                    this.timesteps = timesteps;
                    this.stateful = stateful;
                    this.tanhActivationFunction = new HyperbolicTangent();
                    this.sigmoidActivationFunction = new Sigmoid();

                    for (int i = 0; i < length2; i++)
                    {
                        this.weights[i] = func(inputs, hiddens);
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
                        xWeights[i] = this.weights![i];
                    }

                    for (int i = 0, j = length2; i < length3; i++, j++)
                    {
                        hWeights[i] = this.weights![j];
                    }

                    for (int i = 0; i < inputs.Size; i++)
                    {
                        outputs[i] = new double[length4];
                    }

                    this.layerList = new List<GRUCell>();

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
                        var layer = new GRUCell(this.inputs, this.outputs, xWeights, hWeights, this.biases!, this.tanhActivationFunction!, this.sigmoidActivationFunction!);
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

                        var tuple = this.layerList![t].Backward(dh);

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
                    return new Batch<double[]>(this.gradients!);
                }

                public void SetGradients(Func<bool, double, int, double> func)
                {
                    var length = this.outputs * 3;
                    var offset = this.inputs * length + this.outputs * length;

                    foreach (double[] vector in this.gradients!)
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
                        this.weights![i] = func(this.weights[i], gradients[0][i] / gradients.Size);
                    }

                    for (int i = 0, j = length2; i < length3; i++, j++)
                    {
                        this.weights![j] = func(this.weights[j], gradients[0][j] / gradients.Size);
                    }

                    for (int i = 0, j = offset; i < length1; i++, j++)
                    {
                        this.biases![i] = func(this.biases[i], gradients[0][j] / gradients.Size);
                    }
                }

                private class GRUCell
                {
                    private int inputs = 0;
                    private int hiddens = 0;
                    private double[]? xWeights = null;
                    private double[]? hWeights = null;
                    private double[]? biases = null;
                    private IActivationFunction? tanhActivationFunction = null;
                    private IActivationFunction? sigmoidActivationFunction = null;
                    private Tuple<Batch<double[]>, Batch<double[]>, double[][], double[][], double[][]>? cache = null;

                    public GRUCell(int inputs, int hiddens, double[] xWeights, double[] hWeights, double[] biases, IActivationFunction tanhActivationFunction, IActivationFunction sigmoidActivationFunction)
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
                                    sum1 += vector[j] * this.xWeights![length * j + i];
                                }

                                for (int j = 0; j < this.hiddens; j++)
                                {
                                    sum2 += hPrevious[index][j] * this.hWeights![length * j + i];
                                }

                                z[i] = this.sigmoidActivationFunction!.Forward(sum1 + sum2 + this.biases![i]);
                            }

                            for (int i = 0, j = this.hiddens; i < this.hiddens; i++, j++)
                            {
                                double sum1 = 0.0;
                                double sum2 = 0.0;

                                for (int k = 0; k < this.inputs; k++)
                                {
                                    sum1 += vector[k] * this.xWeights![length * k + j];
                                }

                                for (int k = 0; k < this.hiddens; k++)
                                {
                                    sum2 += hPrevious[index][k] * this.hWeights![length * k + j];
                                }

                                r[i] = this.sigmoidActivationFunction!.Forward(sum1 + sum2 + this.biases![j]);
                            }

                            for (int i = 0, j = this.hiddens * 2; i < this.hiddens; i++, j++)
                            {
                                double sum1 = 0.0;
                                double sum2 = 0.0;

                                for (int k = 0; k < this.inputs; k++)
                                {
                                    sum1 += vector[k] * this.xWeights![length * k + j];
                                }

                                for (int k = 0; k < this.hiddens; k++)
                                {
                                    sum2 += hPrevious[index][k] * r[i] * this.hWeights![length * k + j];
                                }

                                hHat[i] = this.tanhActivationFunction!.Forward(sum1 + sum2 + this.biases![j]);
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
                        var x = this.cache!.Item1;
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

                                dt[i] = dhHat[i] * this.tanhActivationFunction!.Backward(hHat[index][i]);
                                dt[j] = (vector[i] * hHat[index][i] - vector[i] * hPrevious[index][i]) * this.sigmoidActivationFunction!.Backward(z[index][i]);
                            }

                            for (int i = 0, j = 0; i < this.hiddens; i++)
                            {
                                double error1 = 0.0;
                                double error2 = 0.0;

                                for (int k = 0, l = this.hiddens; k < this.hiddens; k++, l++)
                                {
                                    var m = this.hiddens + j;

                                    error1 += dt[k] * this.hWeights![j];
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
                                dt[j] = dhr[i] * hPrevious[index][i] * this.sigmoidActivationFunction!.Backward(r[index][i]);
                            }

                            for (int i = 0, j = 0, offset = this.hiddens * 2; i < this.hiddens; i++)
                            {
                                double error = 0.0;

                                for (int k = 0, l = offset; k < this.hiddens; k++, l++)
                                {
                                    var m = offset + j;

                                    error += dt[l] * this.hWeights![m];
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
                                    error += dt[k] * this.xWeights![j];
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
}
