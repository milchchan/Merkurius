using System;

namespace Milk
{
    namespace LossFunctions
    {
        public interface ILossFunction
        {
            double Function(double y, double a);
            double Derivative(double y, double a);
        }
    }
}
