using System;

namespace Merkurius
{
    namespace ActivationFunctions
    {
        public interface IActivationFunction
        {
            double Function(double x);
            double Derivative(double x);
        }
    }
}
