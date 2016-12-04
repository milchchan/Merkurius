using System;

namespace Milk
{
    namespace ActivationFunctions
    {
        public class Sigmoid : IActivationFunction
        {
            public double Function(double[] x, int i)
            {
                return 1.0 / (1.0 + Math.Exp(-x[i]));
            }

            public double Derivative(double[] x, int i)
            {
                return x[i] * (1.0 - x[i]);
            }
        }
    }
}
