using System;

namespace Megalopolis
{
    namespace ActivationFunctions
    {
        public class HyperbolicTangent : IActivationFunction
        {
            public double Function(double x)
            {
                // (e^x - e^-x) / (e^x + e^-x)
                return Math.Tanh(x);
            }

            public double Derivative(double x)
            {
                // 1 - f(x)^2
                return 1.0 - x * x;
            }
        }
    }
}
