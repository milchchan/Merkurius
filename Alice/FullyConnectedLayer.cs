using System;
using Alice.ActivationFunctions;

namespace Alice
{
    public class FullyConnectedLayer
    {
        private double[] activations = null;
        private double dropoutProbability = 1.0;
        private IActivationFunction activationFunction = null;

        public double[] Activations
        {
            get
            {
                return this.activations;
            }
        }

        public double DropoutProbability
        {
            get
            {
                return this.dropoutProbability;
            }
            set
            {
                this.dropoutProbability = value;
            }
        }

        public IActivationFunction ActivationFunction
        {
            get
            {
                return this.activationFunction;
            }
        }

        public FullyConnectedLayer(int nodes, IActivationFunction activationFunction)
        {
            this.activations = new double[nodes];
            this.activationFunction = activationFunction;

            for (int i = 0; i < nodes; i++)
            {
                this.activations[i] = 1.0;
            }
        }
    }
}
