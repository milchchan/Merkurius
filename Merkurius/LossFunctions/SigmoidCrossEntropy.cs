using System;

namespace Merkurius
{
    namespace LossFunctions
    {
        public class SigmoidCrossEntropy : ILossFunction
        {
            public Tuple<double[], double[]> Forward(double[] y, double[] t)
            {
                double[] vector1 = new double[y.Length];
                double[] vector2 = new double[y.Length];

                for (int i = 0; i < y.Length; i++)
                {
                    vector1[i] = SigmoidFunction(y[i]);
                    vector2[i] = -t[i] * Math.Log(vector1[i] + 1e-7, Math.E) - (1.0 - t[i]) * Math.Log(1.0 - t[i] + 1e-7, Math.E);
                }

                return Tuple.Create<double[], double[]>(vector1, vector2);
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
