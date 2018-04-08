using System;

namespace Megalopolis
{
    namespace LossFunctions
    {
        public class SoftmaxCrossEntropy : ILossFunction
        {
            public double Function(double y, double t)
            {
                return -t * Math.Log(y + 1e-7, Math.E);
            }

            public double Derivative(double y, double t)
            {
                return y - t;
            }
        }
    }
}
