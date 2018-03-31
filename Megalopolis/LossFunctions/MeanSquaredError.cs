using System;

namespace Megalopolis
{
    namespace LossFunctions
    {
        // Mean squared error (MSE) for regression
        public class MeanSquaredError : ILossFunction
        {
            public double Function(double y, double t)
            {
                return (y - t) * (y - t) / 2;
            }

            public double Derivative(double y, double t)
            {
                return y - t;
            }
        }
    }
}
