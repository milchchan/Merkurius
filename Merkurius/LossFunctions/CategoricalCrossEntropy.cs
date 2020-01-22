using System;

namespace Merkurius
{
    namespace LossFunctions
    {
        // Cross-entropy loss function for multiclass classification
        public class CategoricalCrossEntropy : ILossFunction
        {
            public double Function(double y, double t)
            {
                return -t * Math.Log(y + 1e-7, Math.E);
            }

            public double Derivative(double y, double t)
            {
                return -t / y;
            }
        }
    }
}
