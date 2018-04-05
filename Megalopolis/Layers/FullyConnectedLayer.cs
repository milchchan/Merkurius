using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Megalopolis.ActivationFunctions;

namespace Megalopolis
{
    namespace Layers
    {
        public class FullyConnectedLayer : Layer
        {
            private IActivationFunction activationFunction = null;
            private Collection<IFilter> filterCollection = null;

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
                this.filterCollection = new Collection<IFilter>();

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
                this.filterCollection = new Collection<IFilter>();

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(i);
                }

                for (int i = 0; i < layer.InputActivations.Length; i++)
                {
                    this.biases[i] = 0;
                }
            }

            public FullyConnectedLayer(int inputs, int outputs, IActivationFunction activationFunction, IEnumerable<IFilter> filters, Func<int, double> func) : base(inputs, outputs)
            {
                var length = inputs * outputs;

                this.weights = new double[length];
                this.biases = new double[outputs];
                this.activationFunction = activationFunction;
                this.filterCollection = new Collection<IFilter>();

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(i);
                }

                for (int i = 0; i < outputs; i++)
                {
                    this.biases[i] = 0;
                }

                foreach (var filter in filters)
                {
                    this.filterCollection.Add(filter);
                }
            }

            public FullyConnectedLayer(int nodes, IActivationFunction activationFunction, IEnumerable<IFilter> filters, Func<int, double> func, Layer layer) : base(nodes, layer)
            {
                var length = nodes * layer.InputActivations.Length;

                this.weights = new double[length];
                this.biases = new double[layer.InputActivations.Length];
                this.activationFunction = activationFunction;
                this.filterCollection = new Collection<IFilter>();

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(i);
                }

                for (int i = 0; i < layer.InputActivations.Length; i++)
                {
                    this.biases[i] = 0;
                }

                foreach (var filter in filters)
                {
                    this.filterCollection.Add(filter);
                }
            }

            public FullyConnectedLayer(FullyConnectedLayer layer) : base(layer)
            {
                this.weights = new double[layer.weights.Length];
                this.biases = new double[layer.biases.Length];
                this.activationFunction = layer.activationFunction;
                this.filterCollection = new Collection<IFilter>();

                for (int i = 0; i < layer.weights.Length; i++)
                {
                    this.weights[i] = layer.weights[i];
                }

                for (int i = 0; i < layer.biases.Length; i++)
                {
                    this.biases[i] = layer.biases[i];
                }

                foreach (var filter in layer.filterCollection)
                {
                    this.filterCollection.Add(filter.Copy());
                }
            }

            public FullyConnectedLayer(FullyConnectedLayer sourceLayer, Layer targetLayer) : base(sourceLayer, targetLayer)
            {
                this.weights = new double[sourceLayer.weights.Length];
                this.biases = new double[sourceLayer.biases.Length];
                this.activationFunction = sourceLayer.activationFunction;
                this.filterCollection = new Collection<IFilter>();

                for (int i = 0; i < sourceLayer.weights.Length; i++)
                {
                    this.weights[i] = sourceLayer.weights[i];
                }

                for (int i = 0; i < sourceLayer.biases.Length; i++)
                {
                    this.biases[i] = sourceLayer.biases[i];
                }

                foreach (var filter in sourceLayer.filterCollection)
                {
                    this.filterCollection.Add(filter.Copy());
                }
            }

            public override void PropagateForward(bool isTraining)
            {
                double[] summations = new double[this.outputActivations.Length];
                double[] tempActivations = new double[this.outputActivations.Length];

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

                for (int i = 0; i < this.outputActivations.Length; i++)
                {
                    tempActivations[i] = this.activationFunction.Function(summations[i]);
                }

                foreach (var filter in this.filterCollection)
                {
                    tempActivations = filter.PropagateForward(isTraining, tempActivations);
                }

                for (int i = 0; i < this.outputActivations.Length; i++)
                {
                    this.outputActivations[i] = tempActivations[i];
                }
            }

            public override IEnumerable<double[]> PropagateBackward(ref double[] deltas, out double[] gradients)
            {
                gradients = new double[this.inputActivations.Length * this.outputActivations.Length];

                if (this.nextLayer == null)
                {
                    var d1 = new double[this.outputActivations.Length];
                    var d2 = new double[this.inputActivations.Length];

                    for (int i = 0; i < this.outputActivations.Length; i++)
                    {
                        d1[i] = this.activationFunction.Derivative(this.outputActivations[i]) * deltas[i];
                    }

                    foreach (var filter in this.filterCollection)
                    {
                        d1 = filter.PropagateBackward(d1);
                    }

                    for (int i = 0, j = 0; i < this.inputActivations.Length; i++)
                    {
                        double error = 0;

                        for (int k = 0; k < this.outputActivations.Length; k++)
                        {
                            error += d1[k] * this.weights[j];
                            gradients[j] = d1[k] * this.inputActivations[i];
                            j++;
                        }

                        d2[i] = error;
                    }

                    return new double[][] { d1, d2 };
                }

                var d = new double[this.inputActivations.Length];

                for (int i = 0; i < this.outputActivations.Length; i++)
                {
                    deltas[i] = this.activationFunction.Derivative(this.outputActivations[i]) * deltas[i];
                }

                foreach (var filter in this.filterCollection)
                {
                    deltas = filter.PropagateBackward(deltas);
                }

                for (int i = 0, j = 0; i < this.inputActivations.Length; i++)
                {
                    double error = 0;

                    for (int k = 0; k < this.outputActivations.Length; k++)
                    {
                        error += deltas[k] * this.weights[j];
                        gradients[j] = deltas[k] * this.inputActivations[i];
                        j++;
                    }

                    d[i] = error;
                }

                return new double[][] { d };
            }

            public override void Update(double[] gradients, double[] deltas, Func<double, double, double> func)
            {
                var length = this.inputActivations.Length * this.outputActivations.Length;

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(this.weights[i], gradients[i]);
                }

                for (int i = 0; i < this.outputActivations.Length; i++)
                {
                    this.biases[i] = func(this.biases[i], deltas[i]);
                }
            }

            public override Layer Copy()
            {
                return new FullyConnectedLayer(this);
            }

            public override Layer Copy(Layer layer)
            {
                return new FullyConnectedLayer(this, layer);
            }
        }
    }
}
