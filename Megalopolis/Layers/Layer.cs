using System;
using System.Collections.Generic;

namespace Megalopolis
{
    namespace Layers
    {
        public abstract class Layer
        {
            protected int inputs = 0;
            protected int outputs = 0;
            protected double[] weights = null;
            protected double[] biases = null;
            protected Layer previousLayer = null;
            protected Layer nextLayer = null;

            public int Inputs
            {
                get
                {
                    return this.inputs;
                }
            }

            public int Outputs
            {
                get
                {
                    return this.outputs;
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
                this.inputs = inputs;
                this.outputs = outputs;
            }

            public Layer(Layer layer, int nodes)
            {
                this.inputs = layer.outputs;
                this.outputs = nodes;

                layer.nextLayer = this;
                this.previousLayer = layer;
            }

            public abstract Batch<double[]> PropagateForward(Batch<double[]> inputs, bool isTraining);
            public abstract Tuple<Batch<double[]>, Batch<double[]>> PropagateBackward(Batch<double[]> inputs, Batch<double[]> outputs, Batch<double[]> deltas);
            public abstract void Update(Batch<double[]> gradients, Func<double, double, double> func);
        }
    }
}
