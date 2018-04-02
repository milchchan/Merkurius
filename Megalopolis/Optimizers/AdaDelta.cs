using System;
using System.Collections.Generic;

namespace Megalopolis
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

            public double Optimize(int i, double w, double dw)
            {
                KeyValuePair<double, double> kvp;
                double e;
                double dx;

                if (this.dictionary.TryGetValue(i, out kvp))
                {
                    e = this.rho * kvp.Key + (1 - kvp.Key) * dw * dw;
                    dx = -Math.Sqrt(kvp.Value + this.epsilon) / Math.Sqrt(e + this.epsilon) * dw;
                    this.dictionary[i] = new KeyValuePair<double, double>(e, this.rho * kvp.Value + (1 - this.rho) * dx * dx);
                }
                else
                {
                    e = dw * dw;
                    dx = -Math.Sqrt(this.epsilon) / Math.Sqrt(e + this.epsilon) * dw;
                    this.dictionary.Add(i, new KeyValuePair<double, double>(e, (1 - this.rho) * dx * dx));
                }

                return w + dx;
            }
        }
    }
}
