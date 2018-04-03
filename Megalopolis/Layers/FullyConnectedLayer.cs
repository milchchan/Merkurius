using System;
using System.Collections.Generic;
using Megalopolis.ActivationFunctions;

namespace Megalopolis
{
    namespace Layers
    {
        public class FullyConnectedLayer : Layer
        {
            //double[,] _weights = null;
            private IActivationFunction activationFunction = null;
            private IEnumerable<IFilter> filters = null;

            public IActivationFunction ActivationFunction
            {
                get
                {
                    return this.activationFunction;
                }
            }

            public FullyConnectedLayer(int inputs, int outputs, IActivationFunction activationFunction, Func<int, double> func) : base(inputs, outputs)
            {
                var length = inputs * outputs;

                this.weights = new double[length];
                this.biases = new double[outputs];
                this.activationFunction = activationFunction;

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(i);
                }

                for (int i = 0; i < outputs; i++)
                {
                    this.biases[i] = 0;
                }
            }

            public FullyConnectedLayer(int nodes, IActivationFunction activationFunction, Func<int, double> func, Layer layer) : base(nodes, layer)
            {
                var length = nodes * layer.InputActivations.Length;

                this.weights = new double[length];
                this.biases = new double[layer.InputActivations.Length];
                this.activationFunction = activationFunction;

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(i);
                }

                for (int i = 0; i < layer.InputActivations.Length; i++)
                {
                    this.biases[i] = 0;
                }
            }

            public FullyConnectedLayer(int inputs, int outputs, IActivationFunction activationFunction, IEnumerable<IFilter> filter, Func<int, double> func) : base(inputs, outputs)
            {
                var length = inputs * outputs;

                this.weights = new double[length];
                this.biases = new double[outputs];
                this.activationFunction = activationFunction;
                this.filters = filter;

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(i);
                }

                for (int i = 0; i < outputs; i++)
                {
                    this.biases[i] = 0;
                }
            }

            public FullyConnectedLayer(int nodes, IActivationFunction activationFunction, IEnumerable<IFilter> filter, Func<int, double> func, Layer layer) : base(nodes, layer)
            {
                var length = nodes * layer.InputActivations.Length;

                this.weights = new double[length];
                this.biases = new double[layer.InputActivations.Length];
                this.activationFunction = activationFunction;
                this.filters = filter;

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(i);
                }

                for (int i = 0; i < layer.InputActivations.Length; i++)
                {
                    this.biases[i] = 0;
                }
            }

            public override void PropagateForward(bool isTraining)
            {
                double[] summations = new double[this.outputActivations.Length];

                for (int i = 0; i < this.outputActivations.Length; i++)
                {
                    double sum = 0;

                    for (int j = 0; j < this.inputActivations.Length; j++)
                    {
                        sum += this.inputActivations[j] * this.weights[this.outputActivations.Length * j + i];
                    }

                    sum += this.biases[i];

                    summations[i] = sum;
                }

                if (this.filters == null)
                {
                    for (int i = 0; i < this.outputActivations.Length; i++)
                    {
                        this.outputActivations[i] = this.activationFunction.Function(summations[i]);
                    }
                }
                else
                {
                    double[] tempActivations = new double[this.outputActivations.Length];

                    for (int i = 0; i < this.outputActivations.Length; i++)
                    {
                        tempActivations[i] = this.activationFunction.Function(summations[i]);
                    }

                    foreach (var filter in this.filters)
                    {
                        tempActivations = filter.PropagateForward(isTraining, tempActivations);
                    }

                    for (int i = 0; i < this.outputActivations.Length; i++)
                    {
                        this.outputActivations[i] = tempActivations[i];
                    }
                }
            }

            public override IEnumerable<double[]> PropagateBackward(ref double[] gradients)
            {
                if (this.nextLayer == null)
                {
                    var g1 = new double[this.outputActivations.Length];
                    var g2 = new double[this.inputActivations.Length];

                    for (int i = 0; i < this.outputActivations.Length; i++)
                    {
                        g1[i] = this.activationFunction.Derivative(this.outputActivations[i]) * gradients[i];
                    }

                    if (this.filters != null)
                    {
                        foreach (var filter in this.filters)
                        {
                            g1 = filter.PropagateBackward(g1);
                        }
                    }

                    for (int i = 0, j = 0; i < this.inputActivations.Length; i++)
                    {
                        double error = 0;

                        for (int k = 0; k < this.outputActivations.Length; k++)
                        {
                            error += g1[k] * this.weights[j];
                            j++;
                        }

                        g2[i] = error;
                    }

                    return new double[][] { g1, g2 };
                }

                var g = new double[this.inputActivations.Length];

                for (int i = 0; i < this.outputActivations.Length; i++)
                {
                    gradients[i] = this.activationFunction.Derivative(this.outputActivations[i]) * gradients[i];
                }

                if (this.filters != null)
                {
                    foreach (var filter in this.filters)
                    {
                        gradients = filter.PropagateBackward(gradients);
                    }
                }

                for (int i = 0, j = 0; i < this.inputActivations.Length; i++)
                {
                    double error = 0;

                    for (int k = 0; k < this.outputActivations.Length; k++)
                    {
                        error += gradients[k] * this.weights[j];
                        j++;
                    }

                    g[i] = error;
                }

                return new double[][] { g };
            }

            public override void Update(double[] gradients, Func<double, double, double> func)
            {
                for (int i = 0, j = 0; i < this.inputActivations.Length; i++)
                {
                    for (int k = 0; k < gradients.Length; k++)
                    {
                        this.weights[j] = func(this.weights[j], gradients[k] * this.inputActivations[i]);
                        j++;
                    }
                }

                for (int i = 0; i < gradients.Length; i++)
                {
                    this.biases[i] = func(this.biases[i], gradients[i]);
                }
            }
        }
    }
}
