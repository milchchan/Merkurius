using System;

namespace Alice
{
    namespace ActivationFunctions
    {
        public class Softmax : IActivationFunction
        {
            public double Function(double[] x, int i)
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

            public double Derivative(double[] x, int i)
            {
                // yi(1 - yi) i = j
                // yiyj i ≠ j
                return x[i] * (1.0 - x[i]);
            }
        }
    }
}
