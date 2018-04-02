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
            private Dictionary<int, double> gDictionary = null;

            public Momentum()
            {
                this.gDictionary = new Dictionary<int, double>();
            }

            public Momentum(double eta, double momentum)
            {
                this.eta = eta;
                this.alpha = momentum;
                this.gDictionary = new Dictionary<int, double>();
            }

            public double Optimize(int i, double w, double dw)
            {
                double g;

                if (this.gDictionary.TryGetValue(i, out g))
                {
                    this.gDictionary[i] = dw;
                }
                else
                {
                    g = 0;
                    this.gDictionary.Add(i, dw);
                }

                return w - this.eta * dw + this.alpha * g;
            }
        }
    }
}
