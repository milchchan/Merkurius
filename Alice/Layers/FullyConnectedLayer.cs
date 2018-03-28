using System;
using Alice.ActivationFunctions;

namespace Alice
{
    namespace Layers
    {
        public class FullyConnectedLayer : Layer
        {
            private IActivationFunction activationFunction = null;

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

            public override void PropagateForward()
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

                for (int i = 0; i < this.nextLayer.Activations.Length; i++)
                {
                    this.nextLayer.Activations[i] = this.activationFunction.Function(summations[i]);
                }
            }

            public override double[] PropagateBackward(double[] gradients)
            {
                var g = new double[this.activations.Length];

                if (this.nextLayer == null)
                {
                    for (int i = 0; i < this.Activations.Length; i++)
                    {
                        g[i] = this.activationFunction.Derivative(this.activations[i]) * gradients[i];
                    }
                }
                else
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

                return g;
            }
        }
    }
}
