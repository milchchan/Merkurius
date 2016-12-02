using System;

namespace Milk
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
