using System;
using System.Runtime.Serialization;

namespace Merkurius
{
    namespace ActivationFunctions
    {
        [DataContract]
        public class Identity : IActivationFunction
        {
            public double Forward(double x)
            {
                return x;
            }

            public double Backward(double x)
            {
                return 1.0;
            }
        }
    }
}
