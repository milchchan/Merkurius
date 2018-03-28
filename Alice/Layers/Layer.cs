using System;

namespace Alice
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

                for (int i = 0; i < nodes; i++)
                {
                    this.activations[i] = 1.0;
                }
            }

            public void Connect(Layer layer, Func<int, int, double> weightFunc, Func<int, double> biasFunc)
            {
                this.weights = new double[this.activations.Length, layer.activations.Length];
                this.biases = new double[layer.activations.Length];

                for (int i = 0; i < this.activations.Length; i++)
                {
                    for (int j = 0; j < layer.activations.Length; j++)
                    {
                        this.weights[i, j] = weightFunc(i, j);
                    }
                }

                for (int i = 0; i < layer.activations.Length; i++)
                {
                    this.biases[i] = biasFunc(i);
                }

                layer.previousLayer = this;
                this.nextLayer = layer;
            }

            public void Connect(Layer layer)
            {
                this.weights = new double[this.activations.Length, layer.activations.Length];
                this.biases = new double[layer.activations.Length];

                for (int i = 0; i < this.activations.Length; i++)
                {
                    for (int j = 0; j < layer.activations.Length; j++)
                    {
                        this.weights[i, j] = 0;
                    }
                }

                for (int i = 0; i < layer.activations.Length; i++)
                {
                    this.biases[i] = 0;
                }

                layer.previousLayer = this;
                this.nextLayer = layer;
            }

            public void Disconnect()
            {
                this.weights = null;
                this.biases = null;

                if (this.nextLayer != null)
                {
                    this.nextLayer.previousLayer = null;
                    this.nextLayer = null;
                }
            }

            public abstract void PropagateForward();
            public abstract double[] PropagateBackward(double[] gradients);
        }
    }
}
