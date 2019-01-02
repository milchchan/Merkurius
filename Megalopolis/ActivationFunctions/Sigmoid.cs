using System;
using System.Runtime.Serialization;

namespace Megalopolis
{
    namespace ActivationFunctions
    {
        [DataContract]
        public class Sigmoid : IActivationFunction
        {
            public double Function(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }

            public double Derivative(double x)
            {
                // f(x) * (1 - f(x))
                return x * (1.0 - x);
            }
        }
    }
}
