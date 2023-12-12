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
        // Long short-term memory (LSTM)
        [DataContract]
        public class LSTM : Layer, IUpdatable
        {
            [DataMember]
            private LSTMCore? forwardLstm = null;
            [DataMember]
            private LSTMCore? backwardLstm = null;
            [DataMember]
            private double[]? weights = null;
            [DataMember]
            private double[]? biases = null;
            private Batch<double[]>? hiddenState = null;
            private Batch<double[]>? cellState = null;
            private Batch<double[]>? deltaHiddenState = null;

            public double[] Weights
            {
                get
                {
                    if (this.backwardLstm == null)
                    {
                        return this.forwardLstm!.Weights;
                    }

                    return this.weights!;
                }
                set
                {
                    if (this.backwardLstm == null)
                    {
                        this.forwardLstm!.Weights = value;
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
                    if (this.backwardLstm == null)
                    {
                        return this.forwardLstm!.Biases;
                    }

                    return this.biases!;
                }
                set
                {
                    if (this.backwardLstm == null)
                    {
                        this.forwardLstm!.Biases = value;
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
                    return this.forwardLstm!.Timesteps;
                }
            }

            public Batch<double[]>? State
            {
                get
                {
                    if (this.backwardLstm == null)
                    {
                        return this.forwardLstm!.State;
                    }

                    return this.hiddenState;
                }
                set
                {
                    if (this.backwardLstm == null)
                    {
                        this.forwardLstm!.State = value;
                    }

                    this.hiddenState = value;
                }
            }

            public Batch<double[]>? Memory
            {
                get
                {
                    if (this.backwardLstm == null)
                    {
                        return this.forwardLstm!.Memory;
                    }

                    return this.cellState;
                }
                set
                {
                    if (this.backwardLstm == null)
                    {
                        this.forwardLstm!.Memory = value;
                    }

                    this.cellState = value;
                }
            }

            public Batch<double[]>? DeltaState
            {
                get
                {
                    return this.deltaHiddenState;
                }
            }

            public LSTM(int inputs, int hiddens, int timesteps, bool stateful, bool bidirectional, Func<int, int, double> func) : base(inputs, hiddens)
            {
                var length1 = hiddens * 4;
                var length2 = inputs * length1 + hiddens * length1;

                this.forwardLstm = new LSTMCore(inputs, hiddens, timesteps, stateful, func);

                if (bidirectional)
                {
                    var length3 = length1 * 2;

                    this.backwardLstm = new LSTMCore(inputs, hiddens, timesteps, stateful, func);
                    this.weights = new double[length2 * 2];
                    this.biases = new double[length3];

                    for (int i = 0, j = length2; i < length2; i++, j++)
                    {
                        this.weights[i] = this.forwardLstm.Weights[i];
                        this.weights[j] = this.backwardLstm.Weights[i];
                    }

                    for (int i = 0; i < length3; i++)
                    {
                        this.biases[i] = 0.0;
                    }
                }
            }

            public LSTM(int inputs, int hiddens, int timesteps, bool stateful, bool bidirectional, Func<int, int, double> func, Layer layer) : base(inputs, layer)
            {
                var length1 = hiddens * 4;
                var length2 = inputs * length1 + hiddens * length1;

                this.forwardLstm = new LSTMCore(inputs, hiddens, timesteps, stateful, func);

                if (bidirectional)
                {
                    var length3 = length1 * 2;

                    this.backwardLstm = new LSTMCore(inputs, hiddens, timesteps, stateful, func);
                    this.weights = new double[length2 * 2];
                    this.biases = new double[length3];

                    for (int i = 0, j = length2; i < length2; i++, j++)
                    {
                        this.weights[i] = this.forwardLstm.Weights[i];
                        this.weights[j] = this.backwardLstm.Weights[i];
                    }

                    for (int i = 0; i < length3; i++)
                    {
                        this.biases[i] = 0.0;
                    }
                }
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                if (this.backwardLstm == null)
                {
                    return this.forwardLstm!.Forward(inputs, isTraining);
                }

                for (int i = 0, length = this.weights!.Length / 2; i < length; i++)
                {
                    this.forwardLstm!.Weights[i] = this.weights[i];
                    this.backwardLstm.Weights[i] = this.weights[i + length];
                }

                for (int i = 0, length = this.biases!.Length / 2; i < length; i++)
                {
                    this.forwardLstm!.Biases[i] = this.biases[i];
                    this.backwardLstm.Biases[i] = this.biases[i + length];
                }

                if (this.hiddenState != null)
                {
                    for (int i = 0; i < this.hiddenState.Size; i++)
                    {
                        int length = this.hiddenState[i].Length / 2;

                        for (int j = 0; j < length; j++)
                        {
                            this.forwardLstm!.State![i][j] = this.hiddenState[i][j];
                            this.backwardLstm.State![i][j] = this.hiddenState[i][j + length];
                        }
                    }
                }

                if (this.cellState != null)
                {
                    for (int i = 0; i < this.cellState.Size; i++)
                    {
                        int length = this.cellState[i].Length / 2;

                        for (int j = 0; j < length; j++)
                        {
                            this.forwardLstm!.Memory![i][j] = this.cellState[i][j];
                            this.backwardLstm.Memory![i][j] = this.cellState[i][j + length];
                        }
                    }
                }

                var outputs1 = this.forwardLstm!.Forward(inputs, isTraining);
                var outputs2 = this.backwardLstm.Forward(Reverse(inputs), isTraining);
                var vectorList1 = new List<double[]>();
                var vectorList2 = new List<double[]>();
                var vectorList3 = new List<double[]>();

                for (int i = 0; i < outputs1.Size; i++)
                {
                    var vector = new double[outputs1[i].Length];

                    for (int j = 0; j < outputs1[i].Length; j++)
                    {
                        vector[j] = outputs1[i][j] + outputs2[i][j];
                    }

                    vectorList1.Add(vector);
                }

                for (int i = 0; i < this.forwardLstm.State!.Size; i++)
                {
                    int length = this.forwardLstm.State[i].Length;
                    var vector = new double[this.forwardLstm.State[i].Length + this.backwardLstm.State![i].Length];

                    for (int j = 0; j < length; j++)
                    {
                        vector[j] = this.forwardLstm.State[i][j];
                        vector[j + length] = this.backwardLstm.State[i][j];
                    }

                    vectorList2.Add(vector);
                }

                this.hiddenState = new Batch<double[]>(vectorList2);

                for (int i = 0; i < this.forwardLstm.Memory!.Size; i++)
                {
                    int length = this.forwardLstm.Memory[i].Length;
                    var vector = new double[this.forwardLstm.Memory[i].Length + this.backwardLstm.Memory![i].Length];

                    for (int j = 0; j < length; j++)
                    {
                        vector[j] = this.forwardLstm.Memory[i][j];
                        vector[j + length] = this.backwardLstm.Memory[i][j];
                    }

                    vectorList3.Add(vector);
                }

                this.cellState = new Batch<double[]>(vectorList3);

                return new Batch<double[]>(vectorList1);
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                if (this.backwardLstm == null)
                {
                    return this.forwardLstm!.Backward(deltas);
                }

                var dx1 = this.forwardLstm!.Backward(deltas);
                var dx2 = this.backwardLstm.Backward(Reverse(deltas));
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

                this.deltaHiddenState = new Batch<double[]>(new double[this.forwardLstm.DeltaState!.Size][]);

                for (int i = 0; i < this.forwardLstm.DeltaState.Size; i++)
                {
                    this.deltaHiddenState[i] = new double[forwardLstm.DeltaState[i].Length];

                    for (int j = 0; j < this.forwardLstm.DeltaState[i].Length; j++)
                    {
                        this.deltaHiddenState[i][j] = this.forwardLstm.DeltaState[i][j] + this.backwardLstm.DeltaState![i][j];
                    }
                }

                return new Batch<double[]>(vectorList);
            }

            public Batch<double[]> GetGradients()
            {
                if (this.backwardLstm == null)
                {
                    return this.forwardLstm!.GetGradients();
                }

                var gradients1 = this.forwardLstm!.GetGradients();
                var gradients2 = this.backwardLstm.GetGradients();
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
                this.forwardLstm!.SetGradients(func);

                if (this.backwardLstm != null)
                {
                    this.backwardLstm.SetGradients((x, y, z) => func(x, y, this.weights!.Length / 2 - 1 - z));
                }
            }

            public void Update(Batch<double[]> gradients, Func<double, double, double> func)
            {
                if (this.backwardLstm == null)
                {
                    this.forwardLstm!.Update(gradients, func);
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

                    this.forwardLstm!.Update(new Batch<double[]>(vectorList1), func);
                    this.backwardLstm.Update(new Batch<double[]>(vectorList2), func);

                    for (int i = 0, length = this.weights!.Length / 2; i < length; i++)
                    {
                        this.weights[i] = this.forwardLstm.Weights[i];
                        this.weights[i + length] = this.backwardLstm.Weights[i];
                    }

                    for (int i = 0, length = this.biases!.Length / 2; i < length; i++)
                    {
                        this.biases[i] = this.forwardLstm.Biases[i];
                        this.biases[i + length] = this.backwardLstm.Biases[i];
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
            private class LSTMCore : Layer, IUpdatable
            {
                [DataMember]
                private double[]? weights = null;
                [DataMember]
                private double[]? biases = null;
                [DataMember]
                private int timesteps = 0;
                [DataMember]
                private bool stateful = false;
                private Batch<double[]>? h = null; // Hidden state
                private Batch<double[]>? c = null; // Cell state or memory
                private List<LSTMCell>? layerList = null;
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

                public Batch<double[]>? Memory
                {
                    get
                    {
                        return this.c;
                    }
                    set
                    {
                        this.c = value;
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

                public LSTMCore(int inputs, int hiddens, int timesteps, bool stateful, Func<int, int, double> func) : base(inputs, hiddens)
                {
                    var length1 = hiddens * 4;
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
                    var length1 = this.outputs * 4;
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

                    this.layerList = new List<LSTMCell>();

                    if (this.stateful)
                    {
                        if (this.h == null && this.c == null)
                        {
                            this.h = new Batch<double[]>(new double[inputs.Size][]);
                            this.c = new Batch<double[]>(new double[inputs.Size][]);

                            for (int i = 0; i < inputs.Size; i++)
                            {
                                this.h[i] = new double[this.outputs];
                                this.c[i] = new double[this.outputs];

                                for (int j = 0; j < this.outputs; j++)
                                {
                                    this.h[i][j] = 0.0;
                                    this.c[i][j] = 0.0;
                                }
                            }
                        }
                        else
                        {
                            if (this.h == null)
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

                            if (this.c == null)
                            {
                                this.c = new Batch<double[]>(new double[inputs.Size][]);

                                for (int i = 0; i < inputs.Size; i++)
                                {
                                    this.c[i] = new double[this.outputs];

                                    for (int j = 0; j < this.outputs; j++)
                                    {
                                        this.c[i][j] = 0.0;
                                    }
                                }
                            }
                            else if (this.c.Size < inputs.Size)
                            {
                                var batch = new Batch<double[]>(new double[inputs.Size][]);

                                for (int i = 0; i < this.c.Size; i++)
                                {
                                    batch[i] = this.c[i];
                                }

                                for (int i = this.c.Size; i < inputs.Size; i++)
                                {
                                    batch[i] = new double[this.outputs];

                                    for (int j = 0; j < this.outputs; j++)
                                    {
                                        batch[i][j] = 0.0;
                                    }
                                }

                                this.c = batch;
                            }
                        }
                    }
                    else
                    {
                        this.h = new Batch<double[]>(new double[inputs.Size][]);
                        this.c = new Batch<double[]>(new double[inputs.Size][]);

                        for (int i = 0; i < inputs.Size; i++)
                        {
                            this.h[i] = new double[this.outputs];
                            this.c[i] = new double[this.outputs];

                            for (int j = 0; j < this.outputs; j++)
                            {
                                this.h[i][j] = 0.0;
                                this.c[i][j] = 0.0;
                            }
                        }
                    }

                    for (int t = 0; t < this.timesteps; t++)
                    {
                        var layer = new LSTMCell(this.inputs, this.outputs, xWeights, hWeights, this.biases!, this.tanhActivationFunction!, this.sigmoidActivationFunction!);
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

                        var tuple = layer.Forward(x, this.h, this.c);

                        this.h = tuple.Item1;
                        this.c = tuple.Item2;

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

                        var tuple = this.layerList![t].Backward(dh, dc);

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
                    return new Batch<double[]>(this.gradients!);
                }

                public void SetGradients(Func<bool, double, int, double> func)
                {
                    var length = this.outputs * 4;
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

                private class LSTMCell
                {
                    private int inputs = 0;
                    private int hiddens = 0;
                    private double[]? xWeights = null;
                    private double[]? hWeights = null;
                    private double[]? biases = null;
                    private IActivationFunction? tanhActivationFunction = null;
                    private IActivationFunction? sigmoidActivationFunction = null;
                    private Tuple<Batch<double[]>, Batch<double[]>, Batch<double[]>, Tuple<double[][], double[][], double[][], double[][]>, double[][]>? cache = null;

                    public LSTMCell(int inputs, int hiddens, double[] xWeights, double[] hWeights, double[] biases, IActivationFunction tanhActivationFunction, IActivationFunction sigmoidActivationFunction)
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

                        for (int i = 0; i < hPrevious.Size; i++)
                        {
                            data.Item5[i] = hPrevious[i];
                        }

                        for (int i = 0; i < cPrevious.Size; i++)
                        {
                            data.Item6[i] = cPrevious[i];
                        }

                        parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                        Parallel.ForEach<double[], List<Tuple<long, double[], double[], double[], double[], double[], double[]>>>(x, parallelOptions, () => new List<Tuple<long, double[], double[], double[], double[], double[], double[]>>(), (vector, state, index, local) =>
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
                                    sum += hPrevious[index][k] * this.hWeights![length * k + j];
                                }

                                v[j] = sum;
                            }

                            for (int j = 0; j < this.hiddens; j++)
                            {
                                double sum = 0.0;

                                for (int k = 0; k < this.inputs; k++)
                                {
                                    sum += vector[k] * this.xWeights![j];
                                }

                                f[j] = this.sigmoidActivationFunction!.Forward(sum + v[j] + this.biases![j]);
                            }

                            for (int j = 0, k = this.hiddens; j < this.hiddens; j++, k++)
                            {
                                double sum = 0.0;

                                for (int l = 0; l < this.inputs; l++)
                                {
                                    sum += vector[l] * this.xWeights![k + l];
                                }

                                g[j] = this.tanhActivationFunction!.Forward(sum + v[k] + this.biases![k]);
                            }

                            for (int j = 0, k = this.hiddens * 2; j < this.hiddens; j++, k++)
                            {
                                double sum = 0.0;

                                for (int l = 0; l < this.inputs; l++)
                                {
                                    sum += vector[l] * this.xWeights![k + l];
                                }

                                i[j] = this.sigmoidActivationFunction!.Forward(sum + v[k] + this.biases![k]);
                            }

                            for (int j = 0, k = this.hiddens * 3; j < this.hiddens; j++, k++)
                            {
                                double sum = 0.0;

                                for (int l = 0; l < this.inputs; l++)
                                {
                                    sum += vector[l] * this.xWeights![k + l];
                                }

                                o[j] = this.sigmoidActivationFunction!.Forward(sum + v[k] + this.biases![k]);
                            }

                            for (int j = 0; j < this.hiddens; j++)
                            {
                                cNext[j] = f[j] * cPrevious[index][j] + g[j] + i[j];
                                hNext[j] = o[j] * this.tanhActivationFunction!.Forward(cNext[j]);
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
                        var x = this.cache!.Item1;
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
                            var dWx = new double[this.inputs * length];
                            var dhPrev = new double[this.hiddens];
                            var dx = new double[this.inputs];

                            for (int j = 0, k = this.hiddens, l = this.hiddens * 2, m = this.hiddens * 3; j < this.hiddens; j++, k++, l++, m++)
                            {
                                var tanh = this.tanhActivationFunction!.Forward(cNext[index][j]);
                                var ds = dcNext[index][j] + vector[j] * o[index][j] * this.tanhActivationFunction.Backward(tanh);

                                dcPrevious[j] = ds * f[index][j];
                                dA[j] = ds * cPrevious[index][j] * this.sigmoidActivationFunction!.Backward(i[index][j]); // df
                                dA[k] = ds * i[index][j] * this.tanhActivationFunction.Backward(g[index][j]); // dg
                                dA[l] = ds * g[index][j] * this.sigmoidActivationFunction.Backward(i[index][j]); // di
                                dA[m] = vector[j] * tanh * this.sigmoidActivationFunction.Backward(o[index][j]); // do
                            }

                            for (int j = 0, k = 0; j < this.hiddens; j++)
                            {
                                double error = 0.0;

                                for (int l = 0; l < length; l++)
                                {
                                    error += dA[l] * this.hWeights![k];
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
                                    error += dA[l] * this.xWeights![k];
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
}
