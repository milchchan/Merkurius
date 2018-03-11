using System;

namespace Alice
{
    namespace LossFunctions
    {
        // Cross-entropy loss function for multiclass classification
        public class CategoricalCrossEntropy : ILossFunction
        {
            public double Function(double y, double a)
            {
                return -a * Math.Log(y, Math.E);
            }

            public double Derivative(double y, double a)
            {
                return -a / y;
            }
        }
    }
}
