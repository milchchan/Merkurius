using System;
using Milk.ActivationFunctions;
using Milk.Optimizers;

namespace Milk
{
    public class Layer
    {
        private double[] activations = null;
        private double dropoutProbability = 0.5;
        private IActivationFunction activationFunction = null;
        private IOptimizer optimizer = null;

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

        public IOptimizer Optimizer
        {
            get
            {
                return this.optimizer;
            }
        }

        public Layer(int nodes, IActivationFunction activationFunction, IOptimizer optimizer)
        {
            this.activations = new double[nodes];
            this.activationFunction = activationFunction;
            this.optimizer = optimizer;

            for (int i = 0; i < nodes; i++)
            {
                this.activations[i] = 1.0;
            }
        }
    }
}
