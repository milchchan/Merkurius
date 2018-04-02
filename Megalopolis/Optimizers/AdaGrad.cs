using System;
using System.Collections.Generic;

namespace Megalopolis
{
    namespace Optimizers
    {
        public class AdaGrad : IOptimizer
        {
            private double eta = Math.Pow(10, -2); // Learning rate
            private double epsilon = Math.Pow(10, -8);
            private Dictionary<int, double> rDictionary = null;

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

            public double Optimize(int i, double w, double dw)
            {
                double r;

                if (this.rDictionary.TryGetValue(i, out r))
                {
                    r += dw * dw;
                    this.rDictionary[i] = r;
                }
                else
                {
                    r = dw * dw + this.epsilon;
                    this.rDictionary.Add(i, r);
                }

                return w - this.eta / (Math.Sqrt(r)) * dw;
            }
        }
    }
}
