using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace Merkurius
{
    namespace Layers
    {
        [DataContract]
        public class FullyConnected : Layer, IUpdatable
        {
            [DataMember]
            private double[] weights = null;
            [DataMember]
            private double[] biases = null;
            [DataMember]
            private int sequences = 1;
            private Batch<double[]> internalInputs = null;
            private Batch<double[]> internalOutputs = null;
            private List<Tuple<double[], double[]>> gradientList = null;

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

            public FullyConnected(int inputs, int outputs, Func<int, int, double> func) : base(inputs, outputs)
            {
                var length = inputs * outputs;

                this.weights = new double[length];
                this.biases = new double[outputs];

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(inputs, outputs);
                }

                for (int i = 0; i < outputs; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public FullyConnected(int inputs, int outputs, int sequences, Func<int, int, double> func) : base(inputs, outputs)
            {
                var length1 = sequences * inputs * outputs;
                var length2 = sequences * outputs;

                this.outputs = sequences * outputs;
                this.weights = new double[length1];
                this.biases = new double[length2];
                this.sequences = sequences;

                for (int i = 0; i < length1; i++)
                {
                    this.weights[i] = func(inputs, outputs);
                }

                for (int i = 0; i < length2; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public FullyConnected(int nodes, Func<int, int, double> func, Layer layer) : base(nodes, layer)
            {
                var length = nodes * layer.Inputs;

                this.weights = new double[length];
                this.biases = new double[nodes];

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(nodes, layer.Inputs);
                }

                for (int i = 0; i < nodes; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public FullyConnected(int nodes, int sequences, Func<int, int, double> func, Layer layer) : base(nodes, layer)
            {
                var length = nodes * layer.Inputs;

                this.weights = new double[length];
                this.biases = new double[nodes];
                this.sequences = sequences;

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(nodes, layer.Inputs);
                }

                for (int i = 0; i < nodes; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                var parallelOptions = new ParallelOptions();
                var data = new double[inputs.Size][];
                var hiddens = this.outputs / this.sequences;

                this.internalInputs = inputs;

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(inputs, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector, state, index, local) =>
                {
                    var activations = new double[this.outputs];

                    for (int i = 0; i < this.sequences; i++)
                    {
                        var offset1 = this.inputs * i;
                        var offset2 = this.inputs * hiddens * i;
                        var offset3 = hiddens * i;

                        for (int j = 0; j < hiddens; j++)
                        {
                            double sum = 0.0;

                            for (int k = 0; k < this.inputs; k++)
                            {
                                sum += vector[offset1 + k] * this.weights[offset2 + hiddens * k + j];
                            }

                            activations[offset3 + j] = sum + this.biases[offset3 + j];
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

                this.internalOutputs = new Batch<double[]>(data);

                return this.internalOutputs;
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                var parallelOptions = new ParallelOptions();
                var tuple = Tuple.Create<double[][], double[][]>(new double[deltas.Size][], new double[deltas.Size][]);
                var hiddens = this.outputs / this.sequences;

                this.gradientList = new List<Tuple<double[], double[]>>();

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[], double[]>>>(deltas, parallelOptions, () => new List<Tuple<long, double[], double[]>>(), (vector1, state, index, local) =>
                {
                    var gradients = new double[this.sequences * this.inputs * hiddens];
                    var vector2 = new double[this.sequences * this.inputs];

                    for (int i = 0; i < this.sequences; i++)
                    {
                        var offset1 = hiddens * i;
                        var offset2 = this.inputs * hiddens * i;
                        var offset3 = this.inputs * i;

                        for (int j = 0, k = 0; j < this.inputs; j++)
                        {
                            double error = 0.0;

                            for (int l = 0; l < hiddens; l++)
                            {
                                error += vector1[offset1 + l] * this.weights[offset2 + k];
                                gradients[offset2 + k] = vector1[offset1 + l] * this.internalInputs[index][offset3 + j];
                                k++;
                            }

                            vector2[offset3 + j] = error;
                        }
                    }

                    local.Add(Tuple.Create<long, double[], double[]>(index, vector2, gradients));

                    return local;
                }, (local) =>
                {
                    lock (tuple)
                    {
                        local.ForEach(x =>
                        {
                            tuple.Item1[x.Item1] = x.Item2;
                            tuple.Item2[x.Item1] = x.Item3;
                        });
                    }
                });

                for (int i = 0; i < deltas.Size; i++)
                {
                    this.gradientList.Add(Tuple.Create<double[], double[]>(tuple.Item2[i], deltas[i]));
                }

                return new Batch<double[]>(tuple.Item1);
            }

            public Batch<double[]> GetGradients()
            {
                return new Batch<double[]>(this.gradientList.ConvertAll<double[]>(x => x.Item1.Concat<double>(x.Item2).ToArray<double>()));
            }

            public void SetGradients(Func<bool, double, int, double> func)
            {
                this.gradientList.ForEach(x =>
                {
                    for (int i = 0; i < x.Item1.Length; i++)
                    {
                        x.Item1[i] = func(true, x.Item1[i], i);
                    }

                    for (int i = 0; i < x.Item2.Length; i++)
                    {
                        x.Item2[i] = func(false, x.Item2[i], i);
                    }
                });
            }

            public void Update(Batch<double[]> gradients, Func<double, double, double> func)
            {
                var hiddens = this.outputs / this.sequences;
                var length1 = this.inputs * hiddens;

                for (int i = 1; i < gradients.Size; i++)
                {
                    for (int j = 0; j < this.sequences; j++)
                    {
                        var offset = length1 * j;

                        for (int k = 0; k < length1; k++)
                        {
                            gradients[0][k] += gradients[i][offset + k];
                        }

                        for (int k = 0, l = offset + length1; k < hiddens; k++, l++)
                        {
                            gradients[0][l] += gradients[i][l];
                        }
                    }
                }

                for (int i = 0; i < this.sequences; i++)
                {
                    var offset1 = length1 * i;
                    var offset2 = hiddens * i;

                    for (int j = 0; j < length1; j++)
                    {
                        this.weights[offset1 + j] = func(this.weights[offset1 + j], gradients[0][offset1 + j] / gradients.Size);
                    }

                    for (int j = 0, k = length1 * i + length1; j < hiddens; j++, k++)
                    {
                        this.biases[offset2 + j] = func(this.biases[offset2 + j], gradients[0][k] / gradients.Size);
                    }
                }
            }
        }
    }
}
