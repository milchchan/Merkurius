using System;
using System.Runtime.Serialization;

namespace Merkurius
{
    namespace ActivationFunctions
    {
        [DataContract]
        public class Sigmoid : IActivationFunction
        {
            public double Forward(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }

            public double Backward(double x)
            {
                // f(x) * (1 - f(x))
                return x * (1.0 - x);
            }
        }
    }
}
