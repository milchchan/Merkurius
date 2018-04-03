using System;
using System.Collections.Generic;

namespace Megalopolis
{
    namespace Layers
    {
        public abstract class Layer
        {
            protected double[] inputActivations = null;
            protected double[] outputActivations = null;
            protected double[] weights = null;
            protected double[] biases = null;
            protected Layer previousLayer = null;
            protected Layer nextLayer = null;

            public double[] InputActivations
            {
                get
                {
                    return this.inputActivations;
                }
            }

            public double[] OutputActivations
            {
                get
                {
                    return this.outputActivations;
                }
            }

            public double[] Weights
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

            public Layer(int inputs, int outputs)
            {
                this.inputActivations = new double[inputs];
                this.outputActivations = new double[outputs];
            }

            public Layer(int nodes, Layer layer)
            {
                this.inputActivations = new double[nodes];
                this.outputActivations = layer.inputActivations;

                layer.previousLayer = this;
                this.nextLayer = layer;
            }

            public Layer(Layer layer)
            {
                this.inputActivations = new double[layer.inputActivations.Length];
                this.outputActivations = new double[layer.outputActivations.Length];

                for (int i = 0; i < layer.inputActivations.Length; i++)
                {
                    this.inputActivations[i] = layer.inputActivations[i];
                }

                for (int i = 0; i < layer.outputActivations.Length; i++)
                {
                    this.outputActivations[i] = layer.outputActivations[i];
                }
            }

            public Layer(Layer sourceLayer, Layer targetLayer)
            {
                this.inputActivations = new double[sourceLayer.inputActivations.Length];
                this.outputActivations = new double[sourceLayer.outputActivations.Length];

                targetLayer.previousLayer = this;
                this.nextLayer = targetLayer;

                for (int i = 0; i < sourceLayer.inputActivations.Length; i++)
                {
                    this.inputActivations[i] = sourceLayer.inputActivations[i];
                }

                for (int i = 0; i < sourceLayer.outputActivations.Length; i++)
                {
                    this.outputActivations[i] = sourceLayer.outputActivations[i];
                }
            }

            public abstract void PropagateForward(bool isTraining);
            public abstract IEnumerable<double[]> PropagateBackward(ref double[] gradients);
            public abstract void Update(double[] gradients, Func<double, double, double> func);
        }
    }
}
