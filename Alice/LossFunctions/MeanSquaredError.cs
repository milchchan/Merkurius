using System;

namespace Alice
{
    namespace LossFunctions
    {
        public class MeanSquaredError : ILossFunction
        {
            public double Function(double y, double a)
            {
                return (y - a) * (y - a) / 2;
            }

            public double Derivative(double y, double a)
            {
                return y - a;
            }
        }
    }
}
