using System;
using System.Runtime.Serialization;

namespace Merkurius
{
    namespace ActivationFunctions
    {
        [DataContract]
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
