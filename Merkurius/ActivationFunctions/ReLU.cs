using System;
using System.Runtime.Serialization;

namespace Merkurius
{
    namespace ActivationFunctions
    {
        [DataContract]
        public class ReLU : IActivationFunction
        {
            public double Forward(double x)
            {
                if (x > 0)
                {
                    return x;
                }

                return 0;
            }

            public double Backward(double x)
            {
                if (x > 0)
                {
                    return 1.0;
                }

                return 0;
            }
        }
    }
}
