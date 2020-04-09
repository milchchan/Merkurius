using System;
using System.Threading.Tasks;

namespace Merkurius
{
    namespace LossFunctions
    {
        // Cross-entropy loss function for binary classification
        public class BinaryCrossEntropy : ILossFunction
        {
            public Tuple<double[], double[]> Forward(double[] y, double[] t)
            {
                double[] vector = new double[y.Length];

                for (int i = 0; i < y.Length; i++)
                {
                    vector[i] = -t[i] * Math.Log(y[i] + 1e-7, Math.E) - (1.0 - t[i]) * Math.Log(1.0 - t[i] + 1e-7, Math.E);
                }

                return Tuple.Create<double[], double[]>(y, vector);
            }

            public double[] Backward(double[] y, double[] t)
            {
                double[] vector = new double[y.Length];

                for (int i = 0; i < y.Length; i++)
                {
                    vector[i] = (y[i] - t[i]) / (y[i] * (1.0 - y[i]));
                }

                return vector;
            }
        }
    }
}
