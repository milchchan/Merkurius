using System;

namespace Merkurius
{
    namespace LossFunctions
    {
        public class SoftmaxCrossEntropy : ILossFunction
        {
            public Tuple<double[], double[]> Forward(double[] y, double[] t)
            {
                double[] vector1 = new double[y.Length];
                double[] vector2 = new double[y.Length];

                for (int i = 0; i < y.Length; i++)
                {
                    vector1[i] = SoftmaxFunction(y, i);
                    vector2[i] = -t[i] * Math.Log(vector1[i] + 1e-7, Math.E);
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

            private double SoftmaxFunction(double[] x, int i)
            {
                double max = 0.0;
                double sum = 0.0;

                for (int j = 0; j < x.Length; j++)
                {
                    if (x[j] > max)
                    {
                        max = x[j];
                    }
                }

                for (int j = 0; j < x.Length; j++)
                {
                    sum += Math.Exp(x[j] - max);
                }

                return Math.Exp(x[i] - max) / sum;
            }
        }
    }
}
