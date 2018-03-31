using System;
using System.Collections.Generic;
using Megalopolis.ActivationFunctions;

namespace Megalopolis
{
    namespace Layers
    {
        public class FullyConnectedLayer : Layer
        {
            private IActivationFunction activationFunction = null;
            private IEnumerable<IFilter> filters = null;

            public IActivationFunction ActivationFunction
            {
                get
                {
                    return this.activationFunction;
                }
            }

            public FullyConnectedLayer(int nodes, IActivationFunction activationFunction) : base(nodes)
            {
                this.activationFunction = activationFunction;
            }

            public FullyConnectedLayer(int nodes, IActivationFunction activationFunction, IEnumerable<IFilter> filter) : base(nodes)
            {
                this.activationFunction = activationFunction;
                this.filters = filter;
            }

            public override void PropagateForward(bool isTraining)
            {
                double[] summations = new double[this.nextLayer.Activations.Length];

                for (int i = 0; i < this.nextLayer.Activations.Length; i++)
                {
                    double sum = 0;

                    for (int j = 0; j < this.activations.Length; j++)
                    {
                        sum += this.activations[j] * this.weights[j, i];
                    }

                    sum += this.biases[i];

                    summations[i] = sum;
                }

                if (this.filters == null)
                {
                    for (int i = 0; i < this.nextLayer.Activations.Length; i++)
                    {
                        this.nextLayer.Activations[i] = this.activationFunction.Function(summations[i]);
                    }
                }
                else
                {
                    double[] tempActivations = new double[this.nextLayer.Activations.Length];

                    for (int i = 0; i < this.nextLayer.Activations.Length; i++)
                    {
                        tempActivations[i] = this.activationFunction.Function(summations[i]);
                    }

                    foreach (var filter in this.filters)
                    {
                        tempActivations = filter.PropagateForward(isTraining, tempActivations);
                    }

                    for (int i = 0; i < this.nextLayer.Activations.Length; i++)
                    {
                        this.nextLayer.Activations[i] = tempActivations[i];
                    }
                }
            }

            public override double[] PropagateBackward(double[] gradients)
            {
                var g = new double[this.activations.Length];

                if (this.nextLayer == null)
                {
                    if (this.filters == null)
                    {
                        for (int i = 0; i < this.activations.Length; i++)
                        {
                            g[i] = this.activationFunction.Derivative(this.activations[i]) * gradients[i];
                        }
                    }
                    else
                    {
                        double[] tempGradients = new double[this.activations.Length];

                        for (int i = 0; i < this.activations.Length; i++)
                        {
                            tempGradients[i] = this.activationFunction.Derivative(this.activations[i]) * gradients[i];
                        }

                        foreach (var filter in this.filters)
                        {
                            tempGradients = filter.PropagateBackward(tempGradients);
                        }

                        for (int i = 0; i < this.activations.Length; i++)
                        {
                            g[i] = tempGradients[i];
                        }
                    }
                }
                else
                {
                    if (this.filters == null)
                    {
                        for (int i = 0; i < this.activations.Length; i++)
                        {
                            double error = 0;

                            for (int j = 0; j < this.nextLayer.Activations.Length; j++)
                            {
                                error += gradients[j] * this.weights[i, j];
                            }

                            g[i] = this.activationFunction.Derivative(this.activations[i]) * error;
                        }
                    }
                    else
                    {
                        double[] tempGradients = new double[this.Activations.Length];

                        for (int i = 0; i < this.activations.Length; i++)
                        {
                            double error = 0;

                            for (int j = 0; j < this.nextLayer.Activations.Length; j++)
                            {
                                error += gradients[j] * this.weights[i, j];
                            }

                            tempGradients[i] = this.activationFunction.Derivative(this.activations[i]) * error;
                        }

                        foreach (var filter in this.filters)
                        {
                            tempGradients = filter.PropagateBackward(tempGradients);
                        }

                        for (int i = 0; i < this.activations.Length; i++)
                        {
                            g[i] = tempGradients[i];
                        }
                    }
                }

                return g;
            }
        }
    }
}
