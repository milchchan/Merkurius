using System;
using System.Runtime.Serialization;

namespace Megalopolis
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
