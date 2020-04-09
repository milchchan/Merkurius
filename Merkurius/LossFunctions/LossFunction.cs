using System;

namespace Merkurius
{
    namespace LossFunctions
    {
        public interface ILossFunction
        {
            Tuple<double[], double[]> Forward(double[] y, double[] t);
            double[] Backward(double[] y, double[] t);
        }
    }
}
