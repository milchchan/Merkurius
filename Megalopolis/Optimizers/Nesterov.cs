using System;
using System.Collections.Generic;

namespace Megalopolis
{
    namespace Optimizers
    {
        public class Nesterov : IOptimizer
        {
            private double lr = 0.01; // Learning rate
            private double momentum = 0.9; // Momentum
            private Dictionary<int, double> dictionary = null;

            public Nesterov()
            {
                this.dictionary = new Dictionary<int, double>();
            }

            public Nesterov(double lr, double momentum)
            {
                this.lr = lr;
                this.momentum = momentum;
                this.dictionary = new Dictionary<int, double>();
            }

            public double Optimize(int index, double weight, double gradient)
            {
                double v;

                if (this.dictionary.TryGetValue(index, out v))
                {
                    v = v * this.momentum - this.lr * gradient;
                    weight = weight + this.momentum * this.momentum * v - (1.0 + this.momentum) * this.lr * gradient;

                    this.dictionary[index] = v;
                }
                else
                {
                    v = this.lr * gradient;
                    weight = weight + this.momentum * this.momentum * v - (1.0 + this.momentum) * this.lr * gradient;

                    this.dictionary.Add(index, v);
                }

                return weight;
            }
        }
    }
}
