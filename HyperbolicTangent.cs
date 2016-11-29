using System;

namespace Milk
{
    namespace ActivationFunctions
    {
        public class HyperbolicTangent : IActivationFunction
        {
            public double Activate(double x)
            {
                return Math.Tanh(x);
            }

            public double Derivative(double x)
            {
                return 1.0 - x * x;
            }
        }
    }
}
