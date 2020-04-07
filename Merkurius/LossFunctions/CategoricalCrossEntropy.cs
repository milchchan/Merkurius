using System;

namespace Merkurius
{
    namespace LossFunctions
    {
        // Cross-entropy loss function for multiclass classification
        public class CategoricalCrossEntropy : ILossFunction
        {
            public double[] Forward(double[] y, double[] t)
            {
                double[] vector = new double[y.Length];

                for (int i = 0; i < y.Length; i++)
                {
                    vector[i] = -t[i] * Math.Log(y[i] + 1e-7, Math.E);
                }

                return vector;
            }

            public double[] Backward(double[] y, double[] t)
            {
                double[] vector = new double[y.Length];

                for (int i = 0; i < y.Length; i++)
                {
                    vector[i] = -t[i] / y[i];
                }

                return vector;
            }
        }
    }
}
