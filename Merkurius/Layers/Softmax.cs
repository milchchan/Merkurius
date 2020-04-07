using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace Merkurius
{
    namespace Layers
    {
        [DataContract]
        public class Softmax : Layer
        {
            [DataMember]
            private int sequences = 1;
            private Batch<double[]> internalOutputs = null;
            
            public Softmax(int nodes) : base(nodes, nodes) { }

            public Softmax(int nodes, int sequences) : base(nodes, nodes)
            {
                this.sequences = sequences;
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                var parallelOptions = new ParallelOptions();
                var data = new double[inputs.Size][];
                var nodes = this.inputs / this.sequences;
                
                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(inputs, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector, state, index, local) =>
                {
                    double[] activations = new double[this.outputs];

                    for (int i = 0, j = 0; i < this.sequences; i++, j += nodes)
                    {
                        double[] v = new double[nodes];

                        for (int k = 0; k < nodes; k++)
                        {
                            v[k] = vector[j + k];
                        }

                        for (int k = 0; k < nodes; k++)
                        {
                            activations[j + k] = SoftmaxFunction(v, k);
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

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(deltas, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector1, state, index, local) =>
                {
                    var vector2 = new double[this.sequences * this.inputs];

                    for (int i = 0; i < this.sequences; i++)
                    {
                        var offset = hiddens * i;
                        double sum = 0.0;

                        for (int j = 0; j < hiddens; j++)
                        {
                            var dx = this.internalOutputs[index][offset + j] * vector1[offset + j];

                            vector2[offset + j] = dx;
                            sum += dx;
                        }

                        for (int j = 0; j < hiddens; j++)
                        {
                            vector2[offset + j] -= this.internalOutputs[index][offset + j] * sum;
                        }
                    }

                    local.Add(Tuple.Create<long, double[]>(index, vector2));

                    return local;
                }, (local) =>
                {
                    lock (tuple)
                    {
                        local.ForEach(x =>
                        {
                            tuple.Item1[x.Item1] = x.Item2;
                        });
                    }
                });

                return new Batch<double[]>(tuple.Item1);
            }

            private double SoftmaxFunction(double[] x, int i)
            {
                double max = 0.0;
                double sum = 0.0;

                for (int j = 0; j < x.Length; j++)
                {
                    if (x[j] > max)
                    {
                        max = x[j];
                    }
                }

                for (int j = 0; j < x.Length; j++)
                {
                    sum += Math.Exp(x[j] - max);
                }

                return Math.Exp(x[i] - max) / sum;
            }
        }
    }
}
