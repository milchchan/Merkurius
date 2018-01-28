using System;

namespace Alice
{
    namespace LossFunctions
    {
        public class CrossEntropy : ILossFunction
        {
            public double Function(double y, double a)
            {
                return -a * Math.Log(y, Math.E) - (1.0 - a) * Math.Log(1.0 - a, Math.E);
            }

            public double Derivative(double y, double a)
            {
                return (y - a) / (y * (1.0 - y));
            }
        }
    }
}
