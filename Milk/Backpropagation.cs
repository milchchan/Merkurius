using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace Milk
{
    public class Backpropagation : ITrainer
    {
        private Random random = null;
        private double errorThreshold = 0.01;

        public double ErrorThreshold
        {
            get
            {
                return this.errorThreshold;
            }
            set
            {
                this.errorThreshold = value;
            }
        }

        public Backpropagation(Random random)
        {
            this.random = random;
        }

        public void Train(Collection<Layer> layerCollection, Collection<double[,]> weightsCollection, IDictionary<double[], IEnumerable<double[]>> dictionary, int epochs)
        {
            // Backpropagation
        }
    }
}
