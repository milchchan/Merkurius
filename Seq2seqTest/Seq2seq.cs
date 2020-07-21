using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;

namespace Merkurius
{
    namespace Layers
    {
        [DataContract]
        public class Seq2seq : Layer, IUpdatable
        {
            [DataMember]
            private Encoder encoder = null;
            [DataMember]
            private Decoder decoder = null;
            [DataMember]
            private double[] weights = null;

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

            public Seq2seq(int sequenceLength, int vocabularySize, int wordVectorSize, int hiddenSize = 256) : base(sequenceLength, sequenceLength * vocabularySize)
            {
                this.encoder = new Encoder(sequenceLength, vocabularySize, wordVectorSize, hiddenSize);
                this.decoder = new Decoder(sequenceLength, vocabularySize, wordVectorSize, hiddenSize);
                this.weights = new double[this.encoder.Weights.Length + this.decoder.Weights.Length];

                for (int i = 0; i < this.encoder.Weights.Length; i++)
                {
                    this.weights[i] = this.encoder.Weights[i];
                }

                for (int i = 0, j = this.encoder.Weights.Length; i < this.decoder.Weights.Length; i++, j++)
                {
                    this.weights[j] = this.decoder.Weights[i];
                }
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                var vectorList1 = new List<double[]>();
                var vectorList2 = new List<double[]>();

                for (int i = 0; i < inputs.Size; i++)
                {
                    int length = inputs[i].Length / 2;
                    var vector1 = new double[length];
                    var vector2 = new double[length];

                    for (int j = 0; j < length; j++)
                    {
                        vector1[j] = inputs[i][j];
                    }

                    for (int j = length, k = 0; j < inputs[i].Length; j++, k++)
                    {
                        vector2[k] = inputs[i][j];
                    }

                    vectorList1.Add(vector1);
                    vectorList2.Add(vector2);
                }

                for (int i = 0; i < this.encoder.Weights.Length; i++)
                {
                    this.encoder.Weights[i] = this.weights[i];
                }

                for (int i = 0, j = this.encoder.Weights.Length; i < this.decoder.Weights.Length; i++, j++)
                {
                    this.decoder.Weights[i] = this.weights[j];
                }

                this.decoder.EncoderOutputs = this.encoder.Forward(new Batch<double[]>(vectorList1), isTraining);
                this.decoder.State = this.encoder.State;

                return this.decoder.Forward(new Batch<double[]>(vectorList2), isTraining);
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                return this.encoder.Backward(this.decoder.Backward(deltas));
            }

            public Batch<double[]> GetGradients()
            {
                var vectorList = new List<double[]>();
                var encoderGradients = this.encoder.GetGradients();
                var decoderGradients = this.decoder.GetGradients();

                for (int i = 0; i < encoderGradients.Size; i++)
                {
                    vectorList.Add(encoderGradients[i].Concat<double>(decoderGradients[i]).ToArray<double>());
                }

                return new Batch<double[]>(vectorList);
            }

            public void SetGradients(Func<bool, double, int, double> func)
            {
                this.encoder.SetGradients(func);
                this.decoder.SetGradients(func);
            }

            public void Update(Batch<double[]> gradients, Func<double, double, double> func)
            {
                var vectorList1 = new List<double[]>();
                var vectorList2 = new List<double[]>();

                for (int i = 0; i < gradients.Size; i++)
                {
                    var vector1 = new double[this.encoder.Weights.Length];
                    var vector2 = new double[gradients[i].Length - this.encoder.Weights.Length];

                    for (int j = 0; j < this.encoder.Weights.Length; j++)
                    {
                        vector1[j] = gradients[i][j];
                    }

                    for (int j = this.encoder.Weights.Length, k = 0; j < gradients[i].Length; j++, k++)
                    {
                        vector2[k] = gradients[i][j];
                    }

                    vectorList1.Add(vector1);
                    vectorList2.Add(vector2);
                }

                this.encoder.Update(new Batch<double[]>(vectorList1), func);
                this.decoder.Update(new Batch<double[]>(vectorList2), func);

                for (int i = 0; i < this.encoder.Weights.Length; i++)
                {
                    this.weights[i] = this.encoder.Weights[i];
                }

                for (int i = 0, j = this.encoder.Weights.Length; i < this.decoder.Weights.Length; i++, j++)
                {
                    this.weights[j] = this.decoder.Weights[i];
                }
            }
        }
    }
}
