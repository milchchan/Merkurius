using System;

namespace Milk
{
    namespace ActivationFunctions
    {
        public interface IActivationFunction
        {
            double Function(double[] x, int i);
            double Derivative(double[] x, int i);
        }
    }
}
