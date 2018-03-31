using System;

namespace Megalopolis
{
    namespace ActivationFunctions
    {
        public class Identity : IActivationFunction
        {
            public double Function(double x)
            {
                return x;
            }

            public double Derivative(double x)
            {
                return 1.0;
            }
        }
    }
}
