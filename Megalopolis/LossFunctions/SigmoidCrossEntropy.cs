using System;

namespace Megalopolis
{
    namespace LossFunctions
    {
        public class SigmoidCrossEntropy : ILossFunction
        {
            public double Function(double y, double t)
            {
                return -t * Math.Log(y + 1e-7, Math.E) - (1.0 - t) * Math.Log(1.0 - t + 1e-7, Math.E);
            }

            public double Derivative(double y, double t)
            {
                return y - t;
            }
        }
    }
}
