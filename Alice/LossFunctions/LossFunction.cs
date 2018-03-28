using System;

namespace Alice
{
    namespace LossFunctions
    {
        public interface ILossFunction
        {
            double Function(double y, double t);
            double Derivative(double y, double t);
        }
    }
}
