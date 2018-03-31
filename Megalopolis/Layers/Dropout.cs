using System;

namespace Megalopolis
{
    namespace Layers
    {
        public class Dropout : IFilter
        {
            private Random random = null;
            private double rate = 0.5;
            private int[] masks = null;

            public double Rate
            {
                get
                {
                    return this.rate;
                }
            }

            public Dropout(Random random, int nodes)
            {
                this.random = random;
                this.masks = new int[nodes];
            }

            public Dropout(Random random, int nodes, double rate)
            {
                this.random = random;
                this.rate = rate;
                this.masks = new int[nodes];
            }

            public double[] PropagateForward(bool isTraining, double[] activations)
            {
                if (isTraining)
                {
                    for (int i = 0; i < activations.Length; i++)
                    {
                        this.masks[i] = this.random.Binomial(1, this.rate);
                        activations[i] *= this.masks[i];
                    }
                }

                return activations;
            }

            public double[] PropagateBackward(double[] gradients)
            {
                for (int i = 0; i < gradients.Length; i++)
                {
                    gradients[i] *= this.masks[i];
                }

                return gradients;
            }
        }
    }
}
