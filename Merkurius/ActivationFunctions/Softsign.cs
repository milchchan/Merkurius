using System;
using System.Runtime.Serialization;

namespace Merkurius
{
    namespace ActivationFunctions
    {
        [DataContract]
        public class Softsign : IActivationFunction
        {
            public double Forward(double x)
            {
                return x / (1.0 + Math.Abs(x));
            }

            public double Backward(double x)
            {
                return 1.0 / ((1.0 + Math.Abs(x)) * (1.0 + Math.Abs(x)));
            }
        }
    }
}

