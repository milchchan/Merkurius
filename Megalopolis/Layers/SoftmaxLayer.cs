using System;
using System.Collections.Generic;
using Megalopolis.ActivationFunctions;

namespace Megalopolis
{
    namespace Layers
    {
        public class SoftmaxLayer : Layer
        {
            public SoftmaxLayer(int inputs, int outputs, Func<int, double> func) : base(inputs, outputs)
            {
                var length = inputs * outputs;

                this.weights = new double[length];
                this.biases = new double[outputs];

                for (int i = 0; i < length; i++)
                {
                    this.weights[i] = func(i);
                }

                for (int i = 0; i < outputs; i++)
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

                if (isTraining)
                {
                    for (int i = 0; i < this.outputActivations.Length; i++)
                    {
                        this.outputActivations[i] = Softmax(summations, i);
                    }
                }
                else
                {
                    for (int i = 0; i < this.outputActivations.Length; i++)
                    {
                        this.outputActivations[i] = summations[i];
                    }
                }
            }

            public override IEnumerable<double[]> PropagateBackward(ref double[] gradients)
            {
                var g1 = new double[this.outputActivations.Length];
                var g2 = new double[this.inputActivations.Length];

                for (int i = 0; i < this.outputActivations.Length; i++)
                {
                    var vector = DerivativeOfSoftmax(this.outputActivations, i);

                    g1[i] = 0;

                    for (int j = 0; j < this.outputActivations.Length; j++)
                    {
                        g1[i] += vector[j] * gradients[i];
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
