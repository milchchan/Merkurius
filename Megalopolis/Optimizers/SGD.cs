using System;

namespace Megalopolis.Optimizers
{
    public class SGD
    {
        public double lr = 0.001; // Learning rate

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
