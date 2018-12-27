using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using Megalopolis.ActivationFunctions;

namespace Megalopolis
{
    namespace Layers
    {
        public class FullyConnected : Layer, IUpdatable
        {
            private double[] weights = null;
            private double[] biases = null;
            private Batch<double[]> internalInputs = null;
            private Batch<double[]> internalOutputs = null;
            private List<double[]> gradientList = null;
            private IActivationFunction activationFunction = null;
            private Collection<IFilter> filterCollection = null;

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

            public IActivationFunction ActivationFunction
            {
                get
                {
                    return this.activationFunction;
                }
            }

            public FullyConnected(int inputs, int outputs, IActivationFunction activationFunction, Func<int, int, int, double> func) : base(inputs, outputs)
            {
                var length = inputs * outputs;

                this.weights = new double[length];
                this.biases = new double[outputs];
                this.activationFunction = activationFunction;
                this.filterCollection = new Collection<IFilter>();

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(i, inputs, outputs);
                }

                for (int i = 0; i < outputs; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public FullyConnected(Layer layer, int nodes, IActivationFunction activationFunction, Func<int, int, int, double> func) : base(layer, nodes)
            {
                var length = layer.Outputs * nodes;

                this.weights = new double[length];
                this.biases = new double[nodes];
                this.activationFunction = activationFunction;
                this.filterCollection = new Collection<IFilter>();

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(i, layer.Outputs, nodes);
                }

                for (int i = 0; i < nodes; i++)
                {
                    this.biases[i] = 0.0;
                }
            }

            public FullyConnected(int inputs, int outputs, IActivationFunction activationFunction, IEnumerable<IFilter> filters, Func<int, int, int, double> func) : base(inputs, outputs)
            {
                var length = inputs * outputs;

                this.weights = new double[length];
                this.biases = new double[outputs];
                this.activationFunction = activationFunction;
                this.filterCollection = new Collection<IFilter>();

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(i, inputs, outputs);
                }

                for (int i = 0; i < outputs; i++)
                {
                    this.biases[i] = 0.0;
                }

                foreach (var filter in filters)
                {
                    this.filterCollection.Add(filter);
                }
            }

            public FullyConnected(Layer layer, int nodes, IActivationFunction activationFunction, IEnumerable<IFilter> filters, Func<int, int, int, double> func) : base(layer, nodes)
            {
                var length = layer.Outputs * nodes;

                this.weights = new double[length];
                this.biases = new double[nodes];
                this.activationFunction = activationFunction;
                this.filterCollection = new Collection<IFilter>();

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(i, layer.Outputs, nodes);
                }

                for (int i = 0; i < nodes; i++)
                {
                    this.biases[i] = 0.0;
                }

                foreach (var filter in filters)
                {
                    this.filterCollection.Add(filter);
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
                    var activations = new double[this.outputs];

                    for (int i = 0; i < this.outputs; i++)
                    {
                        double sum = 0.0;

                        for (int j = 0; j < this.inputs; j++)
                        {
                            sum += vector[j] * this.weights[this.outputs * j + i];
                        }

                        sum += this.biases[i];

                        activations[i] = this.activationFunction.Function(sum);
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

                foreach (var filter in this.filterCollection)
                {
                    this.internalOutputs = filter.Forward(this.internalOutputs, isTraining);
                }

                return this.internalOutputs;
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                var parallelOptions = new ParallelOptions();
                var data = new double[deltas.Size][];
                var tuple = Tuple.Create<double[][], double[][]>(new double[deltas.Size][], new double[deltas.Size][]);

                this.gradientList = new List<double[]>();

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(deltas, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector1, state, index, local) =>
                {
                    var vector2 = new double[this.outputs];

                    for (int i = 0; i < this.outputs; i++)
                    {
                        vector2[i] = this.activationFunction.Derivative(this.internalOutputs[index][i]) * vector1[i];
                    }

                    local.Add(Tuple.Create<long, double[]>(index, vector2));

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

                var batch = new Batch<double[]>(data);

                foreach (var filter in this.filterCollection)
                {
                    batch = filter.Backward(batch);
                }

                Parallel.ForEach<double[], List<Tuple<long, double[], double[]>>>(batch, parallelOptions, () => new List<Tuple<long, double[], double[]>>(), (vector1, state, index, local) =>
                {
                    var gradients = new double[this.inputs * this.outputs];
                    var vector2 = new double[this.inputs];

                    for (int i = 0, j = 0; i < this.inputs; i++)
                    {
                        double error = 0.0;

                        for (int k = 0; k < this.outputs; k++)
                        {
                            error += vector1[k] * this.weights[j];
                            gradients[j] = vector1[k] * this.internalInputs[index][i];
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
                    this.gradientList.Add(tuple.Item2[i].Concat<double>(batch[i]).ToArray<double>());
                }

                return new Batch<double[]>(tuple.Item1);
            }

            public Batch<double[]> GetGradients()
            {
                return new Batch<double[]>(this.gradientList);
            }

            public void Update(Batch<double[]> gradients, Func<double, double, double> func)
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
        }
    }
}
