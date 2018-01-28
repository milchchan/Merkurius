using System;
using System.Collections.Generic;

namespace Alice
{
    namespace Optimizers
    {
        public class AdaDelta : IOptimizer
        {
            private double rho = 0.95;
            private double epsilon = Math.Pow(10, -8);
            private Dictionary<int, KeyValuePair<double, double>> dictionary = null;

            public AdaDelta()
            {
                this.dictionary = new Dictionary<int, KeyValuePair<double, double>>();
            }

            public AdaDelta(double rho, double epsilon)
            {
                this.rho = rho;
                this.epsilon = rho;
                this.dictionary = new Dictionary<int, KeyValuePair<double, double>>();
            }

            public double Optimize(int index, double weight, double gradient)
            {
                KeyValuePair<double, double> kvp;
                double e;
                double dx;

                if (this.dictionary.TryGetValue(index, out kvp))
                {
                    e = this.rho * kvp.Key + (1 - kvp.Key) * gradient * gradient;
                    dx = -Math.Sqrt(kvp.Value + this.epsilon) / Math.Sqrt(e + this.epsilon) * gradient;
                    this.dictionary[index] = new KeyValuePair<double, double>(e, this.rho * kvp.Value + (1 - this.rho) * dx * dx);
                }
                else
                {
                    e = gradient * gradient;
                    dx = -Math.Sqrt(this.epsilon) / Math.Sqrt(e + this.epsilon) * gradient;
                    this.dictionary.Add(index, new KeyValuePair<double, double>(e, (1 - this.rho) * dx * dx));
                }

                return weight + dx;
            }
        }
    }
}
