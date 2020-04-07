using System;

namespace Merkurius
{
    namespace LossFunctions
    {
        public interface ILossFunction
        {
            double[] Forward(double[] y, double[] t);
            double[] Backward(double[] y, double[] t);
        }
    }
}
