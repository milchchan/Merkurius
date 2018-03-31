using System;

namespace Megalopolis
{
    namespace LossFunctions
    {
        // Cross-entropy loss function for binary classification
        public class BinaryCrossEntropy : ILossFunction
        {
            public double Function(double y, double t)
            {
                return -t * Math.Log(y + 1e-7, Math.E) - (1.0 - t) * Math.Log(1.0 - t + 1e-7, Math.E);
            }

            public double Derivative(double y, double t)
            {
                return (y - t) / (y * (1.0 - y));
            }
        }
    }
}
