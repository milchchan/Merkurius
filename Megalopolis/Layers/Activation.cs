using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Megalopolis.ActivationFunctions;

namespace Megalopolis
{
    namespace Layers
    {
        public class Activation : Layer
        {
            private double[][] internalOutputs = null;
            private IActivationFunction activationFunction = null;

            public IActivationFunction ActivationFunction
            {
                get
                {
                    return this.activationFunction;
                }
            }

            public Activation(Layer layer, IActivationFunction activationFunction) : base(layer, layer.Outputs)
            {
                this.activationFunction = activationFunction;
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                var parallelOptions = new ParallelOptions();

                this.internalOutputs = new double[inputs.Size][];

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(inputs, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector, state, index, local) =>
                {
                    var activations = new double[this.outputs];

                    for (int i = 0; i < this.outputs; i++)
                    {
                        activations[i] = this.activationFunction.Function(vector[i]);
                    }

                    local.Add(Tuple.Create<long, double[]>(index, activations));

                    return local;
                }, (local) =>
                {
                    lock (this.internalOutputs)
                    {
                        local.ForEach(x =>
                        {
                            this.internalOutputs[x.Item1] = x.Item2;
                        });
                    }
                });

                return new Batch<double[]>(this.internalOutputs);
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                var parallelOptions = new ParallelOptions();
                var data = new double[deltas.Size][];
                var tuple = Tuple.Create<double[][], double[][]>(new double[deltas.Size][], new double[deltas.Size][]);

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

                return new Batch<double[]>(data);
            }
        }
    }
}
