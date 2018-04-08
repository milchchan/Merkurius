using System;
using System.Collections.Generic;

namespace Megalopolis
{
    namespace Optimizers
    {
        public class Momentum : IOptimizer
        {
            private double eta = 0.01; // Learning rate
            private double alpha = 0.9; // Momentum
            private Dictionary<int, double> dictionary = null;

            public Momentum()
            {
                this.dictionary = new Dictionary<int, double>();
            }

            public Momentum(double eta, double momentum)
            {
                this.eta = eta;
                this.alpha = momentum;
                this.dictionary = new Dictionary<int, double>();
            }

            public double Optimize(int index, double weight, double gradient)
            {
                double v;

                if (this.dictionary.TryGetValue(index, out v))
                {
                    this.dictionary[index] = gradient;
                }
                else
                {
                    v = 0;
                    this.dictionary.Add(index, gradient);
                }

                return weight - this.eta * gradient + this.alpha * v;
            }
        }
    }
}
