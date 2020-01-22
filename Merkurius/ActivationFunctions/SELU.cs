using System;
using System.Runtime.Serialization;

namespace Merkurius
{
    namespace ActivationFunctions
    {
        [DataContract]
        public class SELU : IActivationFunction
        {
            [DataMember]
            private double alpha = 1.6732632423543772;
            [DataMember]
            private double scale = 1.0507009873554805;

            public SELU() { }

            public SELU(double alpha, double scale)
            {
                this.alpha = alpha;
                this.scale = scale;
            }

            public double Function(double x)
            {
                if (x >= 0)
                {
                    return this.scale * x;
                }

                return this.scale * this.alpha * (Math.Exp(x) - 1.0);
            }

            public double Derivative(double x)
            {
                if (x >= 0)
                {
                    return this.scale;
                }

                return this.scale * this.alpha * Math.Exp(x);
            }
        }
    }
}
