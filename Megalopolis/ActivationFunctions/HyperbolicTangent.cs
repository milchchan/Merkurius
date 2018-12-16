using System;

namespace Megalopolis
{
    namespace ActivationFunctions
    {
        public class HyperbolicTangent : IActivationFunction
        {
            public double Function(double x)
            {
                // (Math.Pow(Math.E, x) - Math.Pow(Math.E, -x)) / (Math.Pow(Math.E, x) + Math.Pow(Math.E, -x))
                return Math.Tanh(x);
            }

            public double Derivative(double x)
            {
                var y = Math.Tanh(x);

                return 1.0 - y * y;
            }
        }
    }
}
