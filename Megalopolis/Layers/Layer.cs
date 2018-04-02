using System;

namespace Megalopolis
{
    namespace Layers
    {
        public abstract class Layer
        {
            protected double[] activations = null;
            protected double[,] weights = null;
            protected double[] biases = null;
            protected Layer previousLayer = null;
            protected Layer nextLayer = null;

            public double[] Activations
            {
                get
                {
                    return this.activations;
                }
            }

            public double[,] Weights
            {
                get
                {
                    return this.weights;
                }
                set
                {
                    this.weights = value;
                }
            }

            public double[] Biases
            {
                get
                {
                    return this.biases;
                }
                set
                {
                    this.biases = value;
                }
            }

            public Layer Previous
            {
                get
                {
                    return this.previousLayer;
                }
            }

            public Layer Next
            {
                get
                {
                    return this.nextLayer;
                }
            }

            public Layer(int nodes)
            {
                this.activations = new double[nodes];
            }

            /*public Layer(int nodes, Layer layer)
            {
                this.activations = new double[nodes];
                layer.previousLayer = this;
                this.nextLayer = layer;
            }*/

            public void Connect(Layer layer)
            {
                layer.previousLayer = this;
                this.nextLayer = layer;
            }

            public void Disconnect()
            {
                if (this.nextLayer != null)
                {
                    this.nextLayer.previousLayer = null;
                    this.nextLayer = null;
                }
            }

            public abstract void PropagateForward(bool isTraining);
            public abstract double[] PropagateBackward(ref double[] gradients);
            public abstract void Update(double[] gradients, Func<double, double, double> func);
        }
    }
}
