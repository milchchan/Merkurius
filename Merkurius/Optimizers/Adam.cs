using System;
using System.Collections.Generic;

namespace Merkurius
{
    namespace Optimizers
    {
        public class Adam : IOptimizer
        {
            private double alpha = 0.001; // Learning rate
            private double beta1 = 0.9; // Decay term
            private double beta2 = 0.999; // Decay term
            private double epsilon = Math.Pow(10, -8);
            private Dictionary<int, ValueTuple<double, double, double, double>> dictionary = null;

            public Adam()
            {
                this.dictionary = new Dictionary<int, ValueTuple<double, double, double, double>>();
            }

            public Adam(double alpha, double beta1, double beta2)
            {
                this.alpha = alpha;
                this.beta1 = beta1;
                this.beta2 = beta2;
                this.dictionary = new Dictionary<int, ValueTuple<double, double, double, double>>();
            }

            public double Optimize(int index, double weight, double gradient)
            {
                ValueTuple<double, double, double, double> tuple;

                if (this.dictionary.TryGetValue(index, out tuple))
                {
                    var mt = this.beta1 * tuple.Item1 + (1.0 - this.beta1) * gradient;
                    var vt = this.beta2 * tuple.Item2 + (1.0 - this.beta2) * (gradient * gradient);

                    weight -= this.alpha * (mt / (1.0 - tuple.Item3)) / Math.Sqrt((vt / (1.0 - tuple.Item4)) + this.epsilon);

                    this.dictionary[index] = ValueTuple.Create<double, double, double, double>(mt, vt, tuple.Item3 * this.beta1, tuple.Item4 * this.beta2);
                }
                else
                {
                    var mt = (1.0 - this.beta1) * gradient;
                    var vt = (1.0 - this.beta2) * (gradient * gradient);

                    weight -= this.alpha * mt / Math.Sqrt(vt + this.epsilon);

                    this.dictionary.Add(index, ValueTuple.Create<double, double, double, double>(mt, vt, this.beta1 * this.beta1, this.beta2 * this.beta2));
                }

                return weight;
            }
        }
    }
}
