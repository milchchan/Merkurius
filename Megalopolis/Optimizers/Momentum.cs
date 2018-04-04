using System;
using System.Collections.Generic;

namespace Megalopolis
{
    namespace Optimizers
    {
        public class Momentum : IOptimizer
        {
            private double eta = 0.5; // Learning rate
            private double alpha = 0.1; // Momentum
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
                double g;

                if (this.dictionary.TryGetValue(index, out g))
                {
                    this.dictionary[index] = gradient;
                }
                else
                {
                    g = 0;
                    this.dictionary.Add(index, gradient);
                }

                return weight - this.eta * gradient + this.alpha * g;
            }
        }
    }
}
