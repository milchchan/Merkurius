using System;

namespace Milk
{
    namespace ActivationFunctions
    {
        public class HyperbolicTangent : IActivationFunction
        {
            public double Function(double[] x, int i)
            {
                return Math.Tanh(x[i]);
            }

            public double Derivative(double[] x, int i)
            {
                return 1.0 - x[i] * x[i];
            }
        }
    }
}
