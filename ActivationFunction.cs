using System;

namespace Milk
{
    namespace ActivationFunctions
    {
        public interface IActivationFunction
        {
            double Activate(double x);
            double Derivative(double x);
        }
    }
}
