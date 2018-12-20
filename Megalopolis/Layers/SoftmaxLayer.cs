using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Megalopolis
{
    namespace Layers
    {
        public class SoftmaxLayer : Layer
        {
            public SoftmaxLayer(Layer layer, int nodes, Func<int, int, int, double> func) : base(layer, nodes)
            {
                var length = layer.Outputs * nodes;

                this.weights = new double[length];
                this.biases = new double[nodes];

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(i, layer.Outputs, nodes);
                }

                for (int i = 0; i < nodes; i++)
                {
                    this.biases[i] = 0;
                }
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                var parallelOptions = new ParallelOptions();
                var data = new double[inputs.Size][];

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(inputs, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector, state, index, local) =>
                {
                    double[] summations = new double[this.outputs];
                    double[] activations = new double[this.outputs];

                    for (int i = 0; i < this.outputs; i++)
                    {
                        double sum = 0;

                        for (int j = 0; j < this.inputs; j++)
                        {
                            sum += vector[j] * this.weights[this.outputs * j + i];
                        }

                        summations[i] = sum + this.biases[i];
                    }

                    for (int i = 0; i < this.outputs; i++)
                    {
                        activations[i] = Softmax(summations, i);
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

            public override Tuple<Batch<double[]>, Batch<double[]>> Backward(Batch<double[]> inputs, Batch<double[]> outputs, Batch<double[]> deltas)
            {
                var parallelOptions = new ParallelOptions();
                var tuple = Tuple.Create<double[][], double[][]>(new double[deltas.Size][], new double[deltas.Size][]);
                List<double[]> vectorList = new List<double[]>();

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[], double[]>>>(deltas, parallelOptions, () => new List<Tuple<long, double[], double[]>>(), (vector1, state, index, local) =>
                {
                    var gradients = new double[this.inputs * this.outputs];
                    var vector2 = new double[this.inputs];

                    for (int i = 0, j = 0; i < this.inputs; i++)
                    {
                        double error = 0;

                        for (int k = 0; k < this.outputs; k++)
                        {
                            error += vector1[k] * this.weights[j];
                            gradients[j] = vector1[k] * inputs[index][i];
                            j++;
                        }

                        vector2[i] = error;
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
                    vectorList.Add(tuple.Item2[i].Concat<double>(deltas[i]).ToArray<double>());
                }

                return Tuple.Create<Batch<double[]>, Batch<double[]>>(new Batch<double[]>(tuple.Item1), new Batch<double[]>(vectorList));
            }

            public override void Update(Batch<double[]> gradients, Func<double, double, double> func)
            {
                var length = this.inputs * this.outputs;

                for (int i = 1; i < gradients.Size; i++)
                {
                    for (int j = 0; j < length; j++)
                    {
                        gradients[0][j] += gradients[i][j];
                    }

                    for (int j = 0, k = length; j < this.outputs; j++, k++)
                    {
                        gradients[0][k] += gradients[i][k];
                    }
                }

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(this.weights[i], gradients[0][i] / gradients.Size);
                }

                for (int i = 0, j = length; i < this.outputs; i++, j++)
                {
                    this.biases[i] = func(this.biases[i], gradients[0][j] / gradients.Size);
                }
            }

            private double Softmax(double[] x, int i)
            {
                double max = 0;
                double sum = 0;

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

            private double[] DerivativeOfSoftmax(double[] x, int i)
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
