using System;

namespace Merkurius
{
    namespace ActivationFunctions
    {
        public interface IActivationFunction
        {
            double Forward(double x);
            double Backward(double x);
        }
    }
}
