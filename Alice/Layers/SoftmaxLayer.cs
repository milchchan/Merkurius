using System;

namespace Alice
{
    namespace Layers
    {
        public class SoftmaxLayer : Layer
        {
            public SoftmaxLayer(int nodes) : base(nodes) { }

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
                    this.nextLayer.Activations[i] = Softmax(summations, i);
                }
            }

            public override double[] PropagateBackward(double[] gradients)
            {
                var g = new double[this.activations.Length];

                for (int i = 0; i < this.activations.Length; i++)
                {
                    var vector = DerivativeOfSoftmax(this.activations, i);

                    g[i] = 0;

                    for (int j = 0; j < this.activations.Length; j++)
                    {
                        g[i] += vector[j] * gradients[i];
                    }
                }

                return g;
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
                // yi(1 - yi) i = j
                // -yiyj i ≠ j
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
