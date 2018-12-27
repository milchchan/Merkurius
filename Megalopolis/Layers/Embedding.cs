using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Megalopolis
{
    namespace Layers
    {
        public class Embedding : Layer, IUpdatable
        {
            private double[] weights = null;
            private Batch<double[]> internalInputs = null;
            private Batch<double[]> dW = null;

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

            public Embedding(int size, int inputs, int outputs, Func<int, int, int, double> func) : base(inputs, outputs)
            {
                this.weights = new double[size];

                for (int i = 0; i < size; i++)
                {
                    this.weights[i] = func(i, inputs, outputs);
                }
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                var parallelOptions = new ParallelOptions();
                var data = new double[inputs.Size][];

                this.internalInputs = inputs;

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(inputs, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector, state, index, local) =>
                {
                    var v = new double[this.inputs * this.outputs];

                    for (int i = 0, j = 0; i < this.inputs; i++)
                    {
                        for (int k = 0, l = this.outputs * Convert.ToInt32(vector[i]); k < this.outputs; k++, l++)
                        {
                            v[j] = this.weights[l];
                            j++;
                        }
                    }

                    local.Add(Tuple.Create<long, double[]>(index, v));

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
                var parallelOptions = new ParallelOptions();
                var data = new double[this.internalInputs.Size][];

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(deltas, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector, state, index, local) =>
                {
                    var dW = new double[this.weights.Length];

                    for (int i = 0; i < this.weights.Length; i++)
                    {
                        dW[i] = 0;
                    }

                    for (int i = 0, j = 0; i < this.inputs; i++)
                    {
                        for (int k = 0, l = this.outputs * Convert.ToInt32(this.internalInputs[index][i]); k < this.outputs; k++, l++)
                        {
                            dW[l] += vector[j];
                            j++;
                        }
                    }

                    local.Add(Tuple.Create<long, double[]>(index, dW));

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

                this.dW = new Batch<double[]>(data);

                return this.dW;
            }

            public Batch<double[]> GetGradients()
            {
                return this.dW;
            }

            public void Update(Batch<double[]> gradients, Func<double, double, double> func)
            {
                for (int i = 1; i < gradients.Size; i++)
                {
                    for (int j = 0; j < this.weights.Length; j++)
                    {
                        gradients[0][j] += gradients[i][j];
                    }
                }

                for (int i = 0; i < this.weights.Length; i++)
                {
                    this.weights[i] = func(this.weights[i], gradients[0][i] / gradients.Size);
                }
            }
        }
    }
}
