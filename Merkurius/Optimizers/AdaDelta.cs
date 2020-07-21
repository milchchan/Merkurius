using System;
using System.Collections.Generic;

namespace Merkurius
{
    namespace Optimizers
    {
        public class AdaDelta : IOptimizer
        {
            private double rho = 0.95;
            private double epsilon = Math.Pow(10, -8);
            private Dictionary<int, ValueTuple<double, double>> dictionary = null;

            public AdaDelta()
            {
                this.dictionary = new Dictionary<int, ValueTuple<double, double>>();
            }

            public AdaDelta(double rho, double epsilon)
            {
                this.rho = rho;
                this.epsilon = rho;
                this.dictionary = new Dictionary<int, ValueTuple<double, double>>();
            }

            public double Optimize(int index, double weight, double gradient)
            {
                ValueTuple<double, double> tuple;
                double e;
                double dx;

                if (this.dictionary.TryGetValue(index, out tuple))
                {
                    e = this.rho * tuple.Item1 + (1 - tuple.Item1) * gradient * gradient;
                    dx = -Math.Sqrt(tuple.Item2 + this.epsilon) / Math.Sqrt(e + this.epsilon) * gradient;
                    this.dictionary[index] = ValueTuple.Create<double, double>(e, this.rho * tuple.Item2 + (1 - this.rho) * dx * dx);
                }
                else
                {
                    e = gradient * gradient;
                    dx = -Math.Sqrt(this.epsilon) / Math.Sqrt(e + this.epsilon) * gradient;
                    this.dictionary.Add(index, ValueTuple.Create<double, double>(e, (1 - this.rho) * dx * dx));
                }

                return weight + dx;
            }
        }
    }
}
