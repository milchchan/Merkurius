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
        public class Softmax : Layer
        {
            [DataMember]
            private int sequences = 1;
            private Batch<double[]> internalInputs = null;
            
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

                this.internalInputs = inputs;

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

                return new Batch<double[]>(data);
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                return deltas;
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

            private double[] DerivativeOfSoftmaxFunction(double[] outputs, double[] deltas)
            {
                double[] dx = new double[deltas.Length];
                double sum = 0.0;

                for (int i = 0; i < deltas.Length; i++)
                {
                    dx[i] = outputs[i] * deltas[i];
                    sum += dx[i];
                }

                for (int i = 0; i < deltas.Length; i++)
                {
                    dx[i] -= outputs[i] * sum;
                }

                return dx;
            }

            private double[] DerivativeOfSoftmaxFunction(double[] x, int i)
            {
                // yi(1 - yi) if i = j
                // -yiyj otherwise
                double[] vector = new double[x.Length];

                for (int j = 0; j < x.Length; j++)
                {
                    if (i == j)
                    {
                        vector[j] = x[i] * (1.0 - x[i]);
                    }
                    else
                    {
                        vector[j] = -x[j] * x[i];
                    }
                }

                return vector;
            }
        }
    }
}
