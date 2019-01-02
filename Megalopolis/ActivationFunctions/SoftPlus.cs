using System;
using System.Runtime.Serialization;

namespace Megalopolis
{
    namespace ActivationFunctions
    {
        [DataContract]
        public class SoftPlus : IActivationFunction
        {
            public double Function(double x)
            {
                // ln(1 + e^x)
                if (x > 0)
                {
                    return x + Math.Log(1.0 + Math.Exp(-x), Math.E);
                }

                return Math.Log(1.0 + Math.Exp(x));
            }

            public double Derivative(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }
        }
    }
}
