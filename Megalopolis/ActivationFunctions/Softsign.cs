using System;

namespace Megalopolis
{
    namespace ActivationFunctions
    {
        public class Softsign : IActivationFunction
        {
            public double Function(double x)
            {
                return x / (1.0 + Math.Abs(x));
            }

            public double Derivative(double x)
            {
                return 1.0 / ((1.0 + Math.Abs(x)) * (1.0 + Math.Abs(x)));
            }
        }
    }
}

