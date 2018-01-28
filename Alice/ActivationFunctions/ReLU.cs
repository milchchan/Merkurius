using System;

namespace Alice
{
    namespace ActivationFunctions
    {
        public class ReLU : IActivationFunction
        {
            public double Function(double[] x, int i)
            {
                if (x[i] > 0)
                {
                    return x[i];
                }

                return 0;
            }

            public double Derivative(double[] x, int i)
            {
                if (x[i] > 0)
                {
                    return 1.0;
                }

                return 0;
            }
        }
    }
}
