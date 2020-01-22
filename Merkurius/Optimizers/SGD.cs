using System;

namespace Merkurius
{
    namespace Optimizers
    {
        public class SGD : IOptimizer
        {
            private double lr = 0.001; // Learning rate

            public SGD() { }

            public SGD(double lr)
            {
                this.lr = lr;
            }

            public double Optimize(int index, double weight, double gradient)
            {
                return weight - this.lr * gradient;
            }
        }
    }
}
