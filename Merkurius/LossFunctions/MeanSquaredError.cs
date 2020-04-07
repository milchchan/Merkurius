using System;

namespace Merkurius
{
    namespace LossFunctions
    {
        // Mean squared error (MSE) for regression
        public class MeanSquaredError : ILossFunction
        {
            public double[] Forward(double[] y, double[] t)
            {
                double[] vector = new double[y.Length];

                for (int i = 0; i < y.Length; i++)
                {
                    vector[i] = (y[i] - t[i]) * (y[i] - t[i]) / 2.0;
                }

                return vector;
            }

            public double[] Backward(double[] y, double[] t)
            {
                double[] vector = new double[y.Length];

                for (int i = 0; i < y.Length; i++)
                {
                    vector[i] = y[i] - t[i];
                }

                return vector;
            }
        }
    }
}
