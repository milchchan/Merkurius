using System;
using System.Runtime.Serialization;

namespace Merkurius
{
    namespace ActivationFunctions
    {
        [DataContract]
        public class SoftPlus : IActivationFunction
        {
            public double Forward(double x)
            {
                // ln(1 + e^x)
                if (x > 0)
                {
                    return x + Math.Log(1.0 + Math.Exp(-x), Math.E);
                }

                return Math.Log(1.0 + Math.Exp(x));
            }

            public double Backward(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }
        }
    }
}
