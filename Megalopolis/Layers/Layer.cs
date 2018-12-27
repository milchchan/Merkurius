using System;

namespace Megalopolis
{
    namespace Layers
    {
        public abstract class Layer
        {
            protected int inputs = 0;
            protected int outputs = 0;
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

            public abstract Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining);
            public abstract Batch<double[]> Backward(Batch<double[]> deltas);
        }
    }
}
