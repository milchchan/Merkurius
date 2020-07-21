using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;

namespace Merkurius
{
    namespace Layers
    {
        [DataContract]
        public class Encoder : Layer, IUpdatable
        {
            [DataMember]
            private Embedding embedding = null;
            [DataMember]
            private LSTM recurrent = null;
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

            public Batch<double[]> State
            {
                get
                {
                    return this.recurrent.State;
                }
                set
                {
                    this.recurrent.State = value;
                }
            }

            public Encoder(int sequenceLength, int vocabularySize, int wordVectorSize, int hiddenSize) : base(sequenceLength, hiddenSize)
            {
                this.embedding = new Embedding(sequenceLength, vocabularySize, wordVectorSize, (fanIn, fanOut) => 0.01 * Initializers.LeCunNormal(fanIn));
                this.recurrent = new LSTM(wordVectorSize, hiddenSize, sequenceLength, false, false, (fanIn, fanOut) => Initializers.LeCunNormal(fanIn));
                this.weights = new double[this.embedding.Weights.Length + this.recurrent.Weights.Length];

                for (int i = 0; i < this.embedding.Weights.Length; i++)
                {
                    this.weights[i] = this.embedding.Weights[i];
                }

                for (int i = 0, j = this.embedding.Weights.Length; i < this.recurrent.Weights.Length; i++, j++)
                {
                    this.weights[j] = this.recurrent.Weights[i];
                }
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                for (int i = 0; i < this.embedding.Weights.Length; i++)
                {
                    this.embedding.Weights[i] = this.weights[i];
                }

                for (int i = 0, j = this.embedding.Weights.Length; i < this.recurrent.Weights.Length; i++, j++)
                {
                    this.recurrent.Weights[i] = this.weights[j];
                }

                return this.recurrent.Forward(this.embedding.Forward(inputs, isTraining), isTraining);
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                return this.embedding.Backward(this.recurrent.Backward(deltas));
            }

            public Batch<double[]> GetGradients()
            {
                var vectorList = new List<double[]>();
                var embeddingGradients = this.embedding.GetGradients();
                var recurrentGradients = this.recurrent.GetGradients();

                for (int i = 0; i < embeddingGradients.Size; i++)
                {
                    vectorList.Add(embeddingGradients[i].Concat<double>(recurrentGradients[i]).ToArray<double>());
                }

                return new Batch<double[]>(vectorList);
            }

            public void SetGradients(Func<bool, double, int, double> func)
            {
                this.embedding.SetGradients(func);
                this.recurrent.SetGradients(func);
            }

            public void Update(Batch<double[]> gradients, Func<double, double, double> func)
            {
                var vectorList1 = new List<double[]>();
                var vectorList2 = new List<double[]>();

                for (int i = 0; i < gradients.Size; i++)
                {
                    var vector1 = new double[this.embedding.Weights.Length];
                    var vector2 = new double[gradients[i].Length - this.embedding.Weights.Length];

                    for (int j = 0; j < this.embedding.Weights.Length; j++)
                    {
                        vector1[j] = gradients[i][j];
                    }

                    for (int j = this.embedding.Weights.Length, k = 0; j < gradients[i].Length; j++, k++)
                    {
                        vector2[k] = gradients[i][j];
                    }

                    vectorList1.Add(vector1);
                    vectorList2.Add(vector2);
                }

                this.embedding.Update(new Batch<double[]>(vectorList1), func);
                this.recurrent.Update(new Batch<double[]>(vectorList2), func);

                for (int i = 0; i < this.embedding.Weights.Length; i++)
                {
                    this.weights[i] = this.embedding.Weights[i];
                }

                for (int i = 0, j = this.embedding.Weights.Length; i < this.recurrent.Weights.Length; i++, j++)
                {
                    this.weights[j] = this.recurrent.Weights[i];
                }
            }
        }
    }
}
