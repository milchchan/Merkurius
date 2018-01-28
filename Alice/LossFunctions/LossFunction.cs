using System;

namespace Alice
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
