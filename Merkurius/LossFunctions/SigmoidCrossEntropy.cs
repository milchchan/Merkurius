using System;

namespace Merkurius
{
    namespace LossFunctions
    {
        public class SigmoidCrossEntropy : ILossFunction
        {
            public double[] Forward(double[] y, double[] t)
            {
                double[] vector = new double[y.Length];

                for (int i = 0; i < y.Length; i++)
                {
                    vector[i] = -t[i] * Math.Log(SigmoidFunction(y[i]) + 1e-7, Math.E) - (1.0 - t[i]) * Math.Log(1.0 - t[i] + 1e-7, Math.E);
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

            private double SigmoidFunction(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }
        }
    }
}
