using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;

namespace Merkurius
{
    namespace Layers
    {
        [DataContract]
        public class Decoder : Layer, IUpdatable
        {
            [DataMember]
            private Embedding? embedding = null;
            [DataMember]
            private LSTM? recurrent = null;
            private Attention? attention = null;
            [DataMember]
            private FullyConnected? fullyConnected = null;
            [DataMember]
            private double[]? weights = null;
            private Batch<double[]>? encoderOutputs = null;

            public double[] Weights
            {
                get
                {
                    return this.weights!;
                }
                set
                {
                    this.weights = value;
                }
            }

            public Batch<double[]>? State
            {
                get
                {
                    return this.recurrent!.State;
                }
                set
                {
                    this.recurrent!.State = value;
                }
            }

            public Batch<double[]>? EncoderOutputs
            {
                get
                {
                    return this.encoderOutputs;
                }
                set
                {
                    this.encoderOutputs = value;
                }
            }

            public Decoder(int sequenceLength, int vocabularySize, int wordVectorSize, int hiddenSize) : base(sequenceLength, sequenceLength * vocabularySize)
            {
                this.embedding = new Embedding(sequenceLength, vocabularySize, wordVectorSize, (fanIn, fanOut) => 0.01 * Initializers.LeCunNormal(fanIn));
                this.recurrent = new LSTM(wordVectorSize, hiddenSize, sequenceLength, true, false, (fanIn, fanOut) => Initializers.LeCunNormal(fanIn));
                this.attention = new Attention(hiddenSize, sequenceLength);
                this.fullyConnected = new FullyConnected(hiddenSize * 2, sequenceLength, sequenceLength * vocabularySize, (fanIn, fanOut) => Initializers.LeCunNormal(fanIn));
                this.weights = new double[this.embedding.Weights.Length + this.recurrent.Weights.Length + this.fullyConnected.Weights.Length];

                for (int i = 0; i < this.embedding.Weights.Length; i++)
                {
                    this.weights[i] = this.embedding.Weights[i];
                }

                for (int i = 0, j = this.embedding.Weights.Length; i < this.recurrent.Weights.Length; i++, j++)
                {
                    this.weights[j] = this.recurrent.Weights[i];
                }

                for (int i = 0, j = this.embedding.Weights.Length + this.recurrent.Weights.Length; i < this.fullyConnected.Weights.Length; i++, j++)
                {
                    this.weights[j] = this.fullyConnected.Weights[i];
                }
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                for (int i = 0; i < this.embedding!.Weights.Length; i++)
                {
                    this.embedding.Weights[i] = this.weights![i];
                }

                for (int i = 0, j = this.embedding.Weights.Length; i < this.recurrent!.Weights.Length; i++, j++)
                {
                    this.recurrent.Weights[i] = this.weights![j];
                }

                for (int i = 0, j = this.embedding.Weights.Length + this.recurrent.Weights.Length; i < this.fullyConnected!.Weights.Length; i++, j++)
                {
                    this.fullyConnected.Weights[i] = this.weights![j];
                }

                var outputs = this.recurrent.Forward(this.embedding.Forward(inputs, isTraining), isTraining);
                
                this.attention!.EncoderOutputs = this.encoderOutputs;

                var contextVectors = this.attention.Forward(outputs, isTraining);
                var outputList = new List<double[]>();

                for (int i = 0; i < outputs.Size; i++)
                {
                    var vector = new double[this.recurrent.Timesteps * outputs[i].Length * 2];

                    for (int j = 0; j < this.recurrent.Timesteps; j++)
                    {
                        for (int k = 0, offset = outputs[i].Length * 2 * j; k < outputs[i].Length; k++)
                        {
                            vector[offset + k] = contextVectors[i][k];
                            vector[offset + k + outputs[i].Length] = outputs[i][k];
                        }
                    }

                    outputList.Add(vector);
                }

                return this.fullyConnected.Forward(new Batch<double[]>(outputList), isTraining);
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                var dout = this.fullyConnected!.Backward(deltas);
                var dcontext = new List<double[]>();
                var dh = new List<double[]>();

                for (int i = 0; i < dout.Size; i++)
                {
                    var length1 = dout[i].Length / 2;
                    var length2 = length1 / this.recurrent!.Timesteps;
                    var length3 = dout[i].Length / this.recurrent.Timesteps;
                    var vector1 = new double[length1];
                    var vector2 = new double[length1];

                    for (int j = 0; j < this.recurrent.Timesteps; j++)
                    {
                        for (int k = 0, offset1 = length2 * j, offset2 = length3 * j; k < length2; k++)
                        {
                            vector1[offset1 + k] = dout[i][offset2 + k];
                            vector2[offset1 + k] = dout[i][offset2 + length2 + k];
                        }
                    }

                    dcontext.Add(vector1);
                    dh.Add(vector2);
                }

                var decoderDeltas = this.attention!.Backward(new Batch<double[]>(dcontext));

                for (int i = 0; i < dh.Count; i++)
                {
                    for (int j = 0; j < dh[i].Length; j++)
                    {
                        decoderDeltas[i][j] += dh[i][j];
                    }
                }

                this.embedding!.Backward(this.recurrent!.Backward(decoderDeltas));

                var encoderDeltaList = new List<double[]>();

                for (int i = 0; i < this.attention!.DeltaEncoderOutputs!.Size; i++)
                {
                    var hiddens = this.attention.DeltaEncoderOutputs[i].Length / this.recurrent.Timesteps;
                    var vector = new double[this.attention.DeltaEncoderOutputs[i].Length];

                    for (int j = 0, k = hiddens * (this.recurrent.Timesteps - 1), l = 0; j < this.attention.DeltaEncoderOutputs[i].Length; j++)
                    {
                        vector[j] = this.attention.DeltaEncoderOutputs[i][j];

                        if (k <= j)
                        {
                            vector[j] += this.recurrent.DeltaState![i][l];
                            l++;
                        }
                    }

                    encoderDeltaList.Add(vector);
                }

                return new Batch<double[]>(encoderDeltaList);
            }

            public Batch<double[]> GetGradients()
            {
                var vectorList = new List<double[]>();
                var embeddingGradients = this.embedding!.GetGradients();
                var recurrentGradients = this.recurrent!.GetGradients();
                var fullyConnectedGradients = this.fullyConnected!.GetGradients();

                for (int i = 0; i < embeddingGradients.Size; i++)
                {
                    vectorList.Add(embeddingGradients[i].Concat<double>(recurrentGradients[i]).Concat<double>(fullyConnectedGradients[i]).ToArray<double>());
                }

                return new Batch<double[]>(vectorList);
            }

            public void SetGradients(Func<bool, double, int, double> func)
            {
                this.embedding!.SetGradients(func);
                this.recurrent!.SetGradients(func);
                this.fullyConnected!.SetGradients(func);
            }

            public void Update(Batch<double[]> gradients, Func<double, double, double> func)
            {
                var vectorList1 = new List<double[]>();
                var vectorList2 = new List<double[]>();
                var vectorList3 = new List<double[]>();

                for (int i = 0; i < gradients.Size; i++)
                {
                    var vector1 = new double[this.embedding!.Weights.Length];
                    var vector2 = new double[gradients[i].Length - this.embedding.Weights.Length - this.fullyConnected!.Weights.Length - this.fullyConnected.Outputs];
                    var vector3 = new double[this.fullyConnected.Weights.Length + this.fullyConnected.Outputs];

                    for (int j = 0; j < this.embedding.Weights.Length; j++)
                    {
                        vector1[j] = gradients[i][j];
                    }

                    for (int j = 0, k = this.embedding.Weights.Length; j < vector2.Length; j++, k++)
                    {
                        vector2[j] = gradients[i][k];
                    }

                    for (int j = 0, k = gradients[i].Length - vector3.Length; j < vector3.Length; j++, k++)
                    {
                        vector3[j] = gradients[i][k];
                    }

                    vectorList1.Add(vector1);
                    vectorList2.Add(vector2);
                    vectorList3.Add(vector3);
                }

                this.embedding!.Update(new Batch<double[]>(vectorList1), func);
                this.recurrent!.Update(new Batch<double[]>(vectorList2), func);
                this.fullyConnected!.Update(new Batch<double[]>(vectorList3), func);

                for (int i = 0; i < this.embedding.Weights.Length; i++)
                {
                    this.weights![i] = this.embedding.Weights[i];
                }

                for (int i = 0, j = this.embedding.Weights.Length; i < this.recurrent.Weights.Length; i++, j++)
                {
                    this.weights![j] = this.recurrent.Weights[i];
                }

                for (int i = 0, j = this.embedding.Weights.Length + this.recurrent.Weights.Length; i < this.fullyConnected.Weights.Length; i++, j++)
                {
                    this.weights![j] = this.fullyConnected.Weights[i];
                }
            }
        }
    }
}
