using System;
using System.Runtime.Serialization;

namespace Merkurius
{
    namespace ActivationFunctions
    {
        [DataContract]
        public class HyperbolicTangent : IActivationFunction
        {
            public double Forward(double x)
            {
                // (e^x - e^-x) / (e^x + e^-x)
                return Math.Tanh(x);
            }

            public double Backward(double x)
            {
                // 1 - f(x)^2
                return 1.0 - x * x;
            }
        }
    }
}
