using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace Merkurius
{
    namespace Layers
    {
        // Attention mechanism
        [DataContract]
        public class Attention : Layer
        {
            [DataMember]
            private int timesteps = 1;
            private Batch<double[]>? encoderOutputs = null;
            private Batch<double[]>? deltaEncoderOutputs = null;
            private List<AttentionCore>? attentionList = null;
            private List<Batch<double[]>>? attentionWeightList = null;

            public Batch<double[]>? EncoderOutputs
            {
                get
                {
                    return this.encoderOutputs;
                }
                set
                {
                    this.encoderOutputs = value;
                }
            }

            public Batch<double[]>? DeltaEncoderOutputs
            {
                get
                {
                    return this.deltaEncoderOutputs;
                }
                set
                {
                    this.deltaEncoderOutputs = value;
                }
            }

            public IEnumerable<Batch<double[]>>? AttentionWeights
            {
                get
                {
                    return this.attentionWeightList;
                }
            }

            public Attention(int nodes) : base(nodes, nodes) { }

            public Attention(int nodes, int sequences) : base(sequences * nodes, sequences * nodes)
            {
                this.timesteps = sequences;
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                var hiddens = this.outputs / this.timesteps;
                var outputs = new double[inputs.Size][];

                for (int i = 0; i < inputs.Size; i++)
                {
                    outputs[i] = new double[this.outputs];
                }

                this.attentionList = new List<AttentionCore>();
                this.attentionWeightList = new List<Batch<double[]>>();

                for (int t = 0; t < this.timesteps; t++)
                {
                    var attentionCore = new AttentionCore(hiddens, this.timesteps);
                    var decoderInputs = new Batch<double[]>(new double[inputs.Size][]);

                    for (int i = 0; i < inputs.Size; i++)
                    {
                        var vector = new double[hiddens];

                        for (int j = 0, k = hiddens * t; j < hiddens; j++, k++)
                        {
                            vector[j] = inputs[i][k];
                        }

                        decoderInputs[i] = vector;
                    }

                    var contextVectors = attentionCore.Forward(this.encoderOutputs!, decoderInputs, isTraining);

                    for (int i = 0; i < inputs.Size; i++)
                    {
                        for (int j = 0, k = hiddens * t; j < hiddens; j++, k++)
                        {
                            outputs[i][k] = contextVectors[i][j];
                        }
                    }

                    this.attentionList.Add(attentionCore);
                }

                return new Batch<double[]>(outputs);
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                var hiddens = this.inputs / this.timesteps;
                var decoderDeltas = new double[deltas.Size][];

                this.deltaEncoderOutputs = new Batch<double[]>(new double[deltas.Size][]);

                for (int i = 0; i < deltas.Size; i++)
                {
                    this.deltaEncoderOutputs[i] = new double[this.inputs];

                    for (int j = 0; j < this.inputs; j++)
                    {
                        this.deltaEncoderOutputs[i][j] = 0.0;
                    }
                }

                for (int t = 0; t < this.timesteps; t++)
                {
                    var d = new Batch<double[]>(new double[deltas.Size][]);

                    for (int i = 0; i < deltas.Size; i++)
                    {
                        var vector = new double[hiddens];

                        for (int j = 0, k = hiddens * t; j < hiddens; j++, k++)
                        {
                            vector[j] = deltas[i][k];
                        }

                        d[i] = vector;
                    }

                    var tuple = this.attentionList![t].Backward(d);

                    for (int i = 0; i < tuple.Item1.Size; i++)
                    {
                        for (int j = 0; j < tuple.Item1[i].Length; j++)
                        {
                            this.deltaEncoderOutputs[i][j] += tuple.Item1[i][j];
                        }
                    }

                    for (int i = 0; i < tuple.Item2.Size; i++)
                    {
                        for (int j = 0, k = hiddens * t; j < hiddens; j++, k++)
                        {
                            decoderDeltas[i][k] = tuple.Item2[i][j];
                        }
                    }
                }

                return new Batch<double[]>(decoderDeltas);
            }

            private class AttentionCore
            {
                private int hiddens = 0;
                private int sequences = 0;
                private Batch<double[]>? internalInputs = null;
                private Batch<double[]>? internalState = null;
                private Softmax? softmax = null;
                private Batch<double[]>? attentionWeight = null;

                public Batch<double[]>? AttentionWeight
                {
                    get
                    {
                        return this.attentionWeight;
                    }
                }

                public AttentionCore(int hiddens, int sequences)
                {
                    this.hiddens = hiddens;
                    this.sequences = sequences;
                    this.softmax = new Softmax(this.sequences);
                }

                public Batch<double[]> Forward(Batch<double[]> inputs, Batch<double[]> h, bool isTraining)
                {
                    var parallelOptions = new ParallelOptions();
                    var data1 = new double[inputs.Size][];
                    var data2 = new double[inputs.Size][];

                    this.internalInputs = inputs;
                    this.internalState = h;

                    parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                    Parallel.ForEach<double[], List<Tuple<long, double[]>>>(inputs, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector1, state, index, local) =>
                    {
                        var vector2 = new double[this.sequences * this.hiddens];
                        var vector3 = new double[this.sequences];

                        for (int i = 0; i < this.sequences; i++)
                        {
                            var offset = this.hiddens * i;

                            for (int j = 0; j < this.hiddens; j++)
                            {
                                vector2[offset + j] = vector1[offset + j] * h[index][j];
                            }
                        }

                        for (int i = 0; i < this.sequences; i++)
                        {
                            vector3[i] = 0.0;

                            for (int j = 0; j < this.hiddens; j++)
                            {
                                vector3[i] += vector2[j];
                            }
                        }

                        local.Add(Tuple.Create<long, double[]>(index, vector3));

                        return local;
                    }, (local) =>
                    {
                        lock (data1)
                        {
                            local.ForEach(x =>
                            {
                                data1[x.Item1] = x.Item2;
                            });
                        }
                    });

                    this.attentionWeight = this.softmax!.Forward(new Batch<double[]>(data1), isTraining);

                    Parallel.ForEach<double[], List<Tuple<long, double[]>>>(inputs, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector1, state, index, local) =>
                    {
                        var vector2 = new double[this.sequences * this.hiddens];
                        var vector3 = new double[this.hiddens];

                        for (int i = 0; i < this.sequences; i++)
                        {
                            var offset = this.hiddens * i;

                            for (int j = 0; j < this.hiddens; j++)
                            {
                                vector2[offset + j] = vector1[offset + j] * this.attentionWeight[index][i];
                            }
                        }

                        for (int i = 0; i < this.hiddens; i++)
                        {
                            vector3[i] = 0.0;

                            for (int j = 0; j < this.sequences; j++)
                            {
                                vector3[i] += vector2[this.hiddens * j + i];
                            }
                        }

                        local.Add(Tuple.Create<long, double[]>(index, vector3));

                        return local;
                    }, (local) =>
                    {
                        lock (data2)
                        {
                            local.ForEach(x =>
                            {
                                data2[x.Item1] = x.Item2;
                            });
                        }
                    });

                    return new Batch<double[]>(data2);
                }

                public Tuple<Batch<double[]>, Batch<double[]>> Backward(Batch<double[]> deltas)
                {
                    var parallelOptions = new ParallelOptions();
                    var tuple1 = Tuple.Create<double[][], double[][]>(new double[deltas.Size][], new double[deltas.Size][]);
                    var tuple2 = Tuple.Create<double[][], double[][]>(new double[deltas.Size][], new double[deltas.Size][]);

                    parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                    Parallel.ForEach<double[], List<Tuple<long, double[], double[]>>>(deltas, parallelOptions, () => new List<Tuple<long, double[], double[]>>(), (vector1, state, index, local) =>
                    {
                        var vector2 = new double[this.sequences * this.hiddens];
                        var vector3 = new double[this.sequences * this.hiddens];
                        var vector4 = new double[this.sequences];

                        for (int i = 0; i < this.sequences; i++)
                        {
                            var offset3 = this.hiddens * i;

                            for (int j = 0; j < this.hiddens; j++)
                            {
                                vector2[offset3 + j] = vector1[j] * this.attentionWeight![index][i];
                                vector3[offset3 + j] = vector1[j] * this.internalInputs![index][offset3 + j];
                            }
                        }

                        for (int i = 0; i < this.sequences; i++)
                        {
                            vector4[i] = 0.0;

                            for (int j = 0; j < this.hiddens; j++)
                            {
                                vector4[i] += vector3[i * this.hiddens + j];
                            }
                        }

                        local.Add(Tuple.Create<long, double[], double[]>(index, vector2, vector4));

                        return local;
                    }, (local) =>
                    {
                        lock (tuple1)
                        {
                            local.ForEach(x =>
                            {
                                tuple1.Item1[x.Item1] = x.Item2;
                                tuple1.Item2[x.Item1] = x.Item3;
                            });
                        }
                    });

                    Parallel.ForEach<double[], List<Tuple<long, double[], double[]>>>(this.softmax!.Backward(new Batch<double[]>(tuple1.Item2)), parallelOptions, () => new List<Tuple<long, double[], double[]>>(), (vector1, state, index, local) =>
                    {
                        var vector2 = new double[this.sequences * this.hiddens];
                        var vector3 = new double[this.sequences * this.hiddens];
                        var vector4 = new double[this.hiddens];

                        for (int i = 0; i < this.sequences; i++)
                        {
                            var offset3 = this.hiddens * i;

                            for (int j = 0; j < this.hiddens; j++)
                            {
                                vector2[offset3 + j] = vector1[i] * this.internalState![index][j];
                                vector3[offset3 + j] = vector1[i] * this.internalInputs![index][offset3 + j];
                            }
                        }

                        for (int i = 0; i < this.hiddens; i++)
                        {
                            vector4[i] = 0.0;

                            for (int j = 0; j < this.sequences; j++)
                            {
                                vector4[i] += vector3[j * this.hiddens + i];
                            }
                        }

                        local.Add(Tuple.Create<long, double[], double[]>(index, vector2, vector4));

                        return local;
                    }, (local) =>
                    {
                        lock (tuple2)
                        {
                            local.ForEach(x =>
                            {
                                tuple2.Item1[x.Item1] = x.Item2;
                                tuple2.Item2[x.Item1] = x.Item3;
                            });
                        }
                    });

                    for (int i = 0; i < deltas.Size; i++)
                    {
                        for (int j = 0; j < tuple1.Item1[i].Length; j++)
                        {
                            tuple1.Item1[i][j] += tuple2.Item1[i][j];
                        }
                    }

                    return Tuple.Create<Batch<double[]>, Batch<double[]>>(new Batch<double[]>(tuple1.Item1), new Batch<double[]>(tuple2.Item2));
                }
            }
        }
    }
}
