using System;
using System.Runtime.Serialization;

namespace Merkurius
{
    namespace Layers
    {
        [DataContract]
        public abstract class Layer
        {
            [DataMember]
            protected int inputs = 0;
            [DataMember]
            protected int outputs = 0;
            protected Layer? previousLayer = null;
            protected Layer? nextLayer = null;

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

            public Layer? Previous
            {
                get
                {
                    return this.previousLayer;
                }
                set
                {
                    this.previousLayer = value;
                }
            }

            public Layer? Next
            {
                get
                {
                    return this.nextLayer;
                }
                set
                {
                    this.nextLayer = value;
                }
            }

            public Layer(int inputs, int outputs)
            {
                this.inputs = inputs;
                this.outputs = outputs;
            }

            public Layer(int nodes, Layer layer)
            {
                this.inputs = nodes;
                this.outputs = layer.inputs;

                layer.Previous = this;
                this.nextLayer = layer;
            }

            public Layer(Layer inputLayer, Layer outputLayer)
            {
                this.inputs = inputLayer.outputs;
                this.outputs = outputLayer.inputs;

                inputLayer.nextLayer = this;
                this.previousLayer = inputLayer;

                outputLayer.Previous = this;
                this.nextLayer = outputLayer;
            }

            public abstract Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining);
            public abstract Batch<double[]> Backward(Batch<double[]> deltas);
        }
    }
}
