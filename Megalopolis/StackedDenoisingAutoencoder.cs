using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Megalopolis.ActivationFunctions;
using Megalopolis.Layers;

namespace Megalopolis
{
    public class StackedDenoisingAutoencoder : ITrainer
    {
        private Random random = null;
        private double learningRate = 0.1;
        private double corruptionLevel = 0.3;
        private IActivationFunction activationFunction = null;

        public double LearningRate
        {
            get
            {
                return this.learningRate;
            }
            set
            {
                this.learningRate = value;
            }
        }

        public double CorruptionLevel
        {
            get
            {
                return this.corruptionLevel;
            }
            set
            {
                this.corruptionLevel = value;
            }
        }

        public StackedDenoisingAutoencoder(Random random, IActivationFunction activationFunction)
        {
            this.random = random;
            this.activationFunction = activationFunction;
        }

        public void Train(Collection<Layer> layerCollection, IDictionary<double[], IEnumerable<double[]>> dictionary, int epochs)
        {
            // Stacked Denoising Autoencoders (SdA)
            var vectorList = dictionary.Values.Aggregate<IEnumerable<double[]>, List<int[]>>(new List<int[]>(), (list, vectors) =>
            {
                foreach (var vector1 in vectors)
                {
                    int[] vector2 = new int[vector1.Length];

                    for (int i = 0; i < vector1.Length; i++)
                    {
                        vector2[i] = (int)vector1[i];
                    }

                    list.Add(vector2);
                }

                return list;
            });
            var inputBiasesList = new List<double[]>();
            var hiddenBiasesList = new List<double[]>();

            for (int i = 1, length = layerCollection.Count - 1; i < length; i++)
            {
                var inputBiases = new double[layerCollection[i - 1].Activations.Length];
                var hiddenBiases = new double[layerCollection[i].Activations.Length];

                for (int j = 0; j < inputBiases.Length; j++)
                {
                    inputBiases[i] = 0;
                }

                inputBiasesList.Add(inputBiases);

                for (int j = 0; j < hiddenBiases.Length; j++)
                {
                    hiddenBiases[i] = 0;
                }

                hiddenBiasesList.Add(hiddenBiases);
            }

            for (int i = 0, length = layerCollection.Count - 2; i < length; i++)
            {
                int t = 0;

                while (t < epochs)
                {
                    foreach (var vector in vectorList)
                    {
                        var inputVector = new int[vector.Length];
                        int j = 1;

                        for (int k = 0; k < vector.Length; k++)
                        {
                            inputVector[k] = vector[k];
                        }

                        while (j <= i)
                        {
                            var tempVector = new int[inputVector.Length];

                            for (int k = 0; k < inputVector.Length; k++)
                            {
                                tempVector[k] = inputVector[k];
                            }

                            inputVector = new int[layerCollection[j].Activations.Length];

                            var summations = new double[layerCollection[j].Activations.Length];

                            for (int k = 0; k < inputVector.Length; k++)
                            {
                                double sum = 0;

                                for (int l = 0; l < tempVector.Length; l++)
                                {
                                    sum += layerCollection[j].Weights[k, l] * tempVector[l];
                                }

                                sum += inputBiasesList[j][k];
                                summations[k] = sum;
                            }

                            for (int k = 0; k < summations.Length; k++)
                            {
                                inputVector[k] = this.random.Binomial(1, this.activationFunction.Function(summations[k]));
                            }

                            j++;
                        }

                        var x = new int[i == 0 ? vector.Length : layerCollection[i].Activations.Length];
                        var y = new double[layerCollection[i + 1].Activations.Length];
                        var z = new double[x.Length];
                        var ySummations = new double[y.Length];
                        var zSummations = new double[z.Length];
                        var tempInputBiases = new double[x.Length];
                        var tempHiddenBiases = new double[y.Length];

                        for (int n = 0; n < x.Length; n++)
                        {
                            if (inputVector[n] == 0)
                            {
                                x[n] = 0;
                            }
                            else
                            {
                                x[n] = this.random.Binomial(1, 1.0 - this.corruptionLevel);
                            }
                        }

                        // Encode
                        for (int n = 0; n < y.Length; n++)
                        {
                            double sum = 0;

                            for (int m = 0; m < x.Length; m++)
                            {
                                sum += layerCollection[i].Weights[m, n] * x[m];
                            }

                            sum += hiddenBiasesList[i][n];
                            ySummations[n] = sum;
                        }

                        for (int n = 0; n < ySummations.Length; n++)
                        {
                            y[n] = this.activationFunction.Function(ySummations[n]);
                        }

                        // Decode
                        for (int n = 0; n < z.Length; n++)
                        {
                            double sum = 0;

                            for (int m = 0; m < y.Length; m++)
                            {
                                sum += layerCollection[i].Weights[n, m] * y[m];
                            }

                            sum += inputBiasesList[i][n];
                            zSummations[n] = sum;
                        }

                        for (int n = 0; n < zSummations.Length; n++)
                        {
                            z[n] = this.activationFunction.Function(zSummations[n]);
                        }

                        for (int n = 0; n < z.Length; n++)
                        {
                            tempInputBiases[n] = inputVector[n] - z[n];
                            inputBiasesList[i][n] += this.learningRate * tempInputBiases[n] / vectorList.Count;
                        }

                        for (int n = 0; n < tempHiddenBiases.Length; n++)
                        {
                            tempHiddenBiases[n] = 0;

                            for (int m = 0; m < tempInputBiases.Length; m++)
                            {
                                tempHiddenBiases[n] += layerCollection[i].Weights[m, n] * tempInputBiases[m];
                            }

                            tempHiddenBiases[n] *= y[n] * (1 - y[n]);
                            hiddenBiasesList[i][n] += this.learningRate * tempHiddenBiases[n] / vectorList.Count;
                        }

                        for (int n = 0; n < layerCollection[i].Activations.Length; n++)
                        {
                            for (int m = 0; m < layerCollection[i + 1].Activations.Length; m++)
                            {
                                if (n < x.Length)
                                {
                                    layerCollection[i].Weights[n, m] += this.learningRate * (tempHiddenBiases[m] * x[n] + tempInputBiases[n] * y[m]) / vectorList.Count;
                                }
                                else
                                {
                                    layerCollection[i].Weights[n, m] = inputBiasesList[i][n];
                                }
                            }
                        }
                    }

                    t++;
                }
            }
        }
    }
}
