using System;
using System.Collections.Generic;

namespace Milk
{
    namespace Optimizers
    {
        public class AdaGrad : IOptimizer
        {
            private readonly double eta = Math.Pow(10, -2); // Learning rate
            private readonly double epsilon = Math.Pow(10, -8);
            private readonly Dictionary<int, double> rDictionary = null;

            public AdaGrad()
            {
                this.rDictionary = new Dictionary<int, double>();
            }

            public AdaGrad(double eta, double epsilon)
            {
                this.eta = eta;
                this.epsilon = epsilon;
                this.rDictionary = new Dictionary<int, double>();
            }

            public double Optimize(int index, double weight, double gradient)
            {
                double r;

                if (this.rDictionary.TryGetValue(index, out r))
                {
                    r += gradient * gradient;
                    this.rDictionary[index] = r;
                }
                else
                {
                    r = gradient * gradient + this.epsilon;
                    this.rDictionary.Add(index, r);
                }

                return weight - this.eta / (Math.Sqrt(r)) * gradient;
            }
        }
    }
}
