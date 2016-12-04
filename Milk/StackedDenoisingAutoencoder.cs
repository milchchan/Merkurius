using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace Milk
{
    public class StackedDenoisingAutoencoder : ITrainer
    {
        private Random random = null;
        private double learningRate = 0.1;
        private double corruptionLevel = 0.3;

        public double LearningRate
        {
            get
            {
                return this.learningRate;
            }
            set
            {
                this.learningRate = value;
            }
        }

        public double CorruptionLevel
        {
            get
            {
                return this.corruptionLevel;
            }
            set
            {
                this.corruptionLevel = value;
            }
        }

        public StackedDenoisingAutoencoder(Random random)
        {
            this.random = random;
        }

        public void Train(Collection<Layer> layerList, Collection<double[,]> weightsCollection, IDictionary<double[], IEnumerable<double[]>> dictionary, int epochs)
        {
            // Stacked Denoising Autoencoders (SdA)
        }
    }
}
