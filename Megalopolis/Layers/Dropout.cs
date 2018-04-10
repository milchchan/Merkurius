using System;
using System.Collections.Generic;

namespace Megalopolis
{
    namespace Layers
    {
        public class Dropout : IFilter
        {
            private Random random = null;
            private double rate = 0.5;
            private Dictionary<int, double> maskDictionary = null;

            public double Rate
            {
                get
                {
                    return this.rate;
                }
            }

            public Dropout()
            {
                this.random = RandomProvider.GetRandom();
                this.maskDictionary = new Dictionary<int, double>();
            }

            public Dropout(double rate)
            {
                this.random = RandomProvider.GetRandom();
                this.rate = rate;
                this.maskDictionary = new Dictionary<int, double>();
            }

            public Dropout(Dropout dropout)
            {
                this.random = RandomProvider.GetRandom();
                this.rate = dropout.rate;
                this.maskDictionary = new Dictionary<int, double>();

                foreach (var keyValuePair in dropout.maskDictionary)
                {
                    this.maskDictionary.Add(keyValuePair.Key, keyValuePair.Value);
                }
            }

            public double[] PropagateForward(bool isTraining, double[] activations)
            {
                if (isTraining)
                {
                    for (int i = 0; i < activations.Length; i++)
                    {
                        double probability = this.random.Binomial(1, this.rate);

                        activations[i] *= probability;

                        if (this.maskDictionary.TryGetValue(i, out probability))
                        {
                            this.maskDictionary[i] = probability;
                        }
                        else
                        {
                            this.maskDictionary.Add(i, probability);
                        }
                    }
                }

                return activations;
            }

            public double[] PropagateBackward(double[] gradients)
            {
                for (int i = 0; i < gradients.Length; i++)
                {
                    gradients[i] *= this.maskDictionary[i];
                }

                return gradients;
            }

            public IFilter Copy()
            {
                return new Dropout(this);
            }
        }
    }
}
