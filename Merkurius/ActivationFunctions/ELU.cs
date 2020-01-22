using System;
using System.Runtime.Serialization;

namespace Merkurius
{
    namespace ActivationFunctions
    {
        [DataContract]
        public class ELU : IActivationFunction
        {
            [DataMember]
            private double alpha = 1.0;

            public ELU() { }

            public ELU(double alpha)
            {
                this.alpha = alpha;
            }

            public double Function(double x)
            {
                // a(e^x - 1) if x < 0
                // x otherwise
                if (x >= 0)
                {
                    return x;
                }

                return this.alpha * (Math.Exp(x) - 1.0);
            }

            public double Derivative(double x)
            {
                // f(a, x) + a if x < 0
                // 1 otherwise
                if (x >= 0)
                {
                    return 1.0;
                }

                return this.alpha * Math.Exp(x);
            }
        }
    }
}
