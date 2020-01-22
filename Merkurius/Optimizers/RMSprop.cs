using System;
using System.Collections.Generic;

namespace Merkurius
{
    namespace Optimizers
    {
        public class RMSprop : IOptimizer
        {
            private double eta = Math.Pow(10, -2); // Learning rate
            private double rho = 0.95;
            private double epsilon = Math.Pow(10, -8);
            private Dictionary<int, double> rDictionary = null;

            public RMSprop()
            {
                this.rDictionary = new Dictionary<int, double>();
            }

            public RMSprop(double eta, double rho, double epsilon)
            {
                this.eta = eta;
                this.rho = rho;
                this.epsilon = epsilon;
                this.rDictionary = new Dictionary<int, double>();
            }

            public double Optimize(int index, double weight, double gradient)
            {
                double r;

                if (this.rDictionary.TryGetValue(index, out r))
                {
                    r += this.rho * r + (1.0 - r) * gradient * gradient;
                    this.rDictionary[index] = r;
                }
                else
                {
                    r = gradient * gradient + this.epsilon;
                    this.rDictionary.Add(index, r);
                }

                return weight - this.eta / Math.Sqrt(r) * gradient;
            }
        }
    }
}
