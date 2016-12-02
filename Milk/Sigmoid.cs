using System;

namespace Milk
{
    namespace ActivationFunctions
    {
        public class Sigmoid : IActivationFunction
        {
            public double Function(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }

            public double Derivative(double x)
            {
                return x * (1.0 - x);
            }
        }
    }
}
