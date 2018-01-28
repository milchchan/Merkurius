using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace Alice
{
    public class StackedDenoisingAutoencoder : ITrainer
    {
        private Random random = null;
        private double learningRate = 0.1;
        private double corruptionLevel = 0.3;

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

        public StackedDenoisingAutoencoder(Random random)
        {
            this.random = random;
        }

        public void Train(Collection<Layer> layerCollection, Collection<double[,]> weightsCollection, IDictionary<double[], IEnumerable<double[]>> dictionary, int epochs)
        {
            // Stacked Denoising Autoencoders (SdA)
            List<int[]> vectorList = dictionary.Values.Aggregate<IEnumerable<double[]>, List<int[]>>(new List<int[]>(), (list, vectors) =>
            {
                foreach (double[] vector1 in vectors)
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
            List<double[]> inputBiasesList = new List<double[]>();
            List<double[]> hiddenBiasesList = new List<double[]>();

            for (int i = 1; i < layerCollection.Count - 1; i++)
            {
                double[] inputBiases = new double[layerCollection[i - 1].Activations.Length];
                double[] hiddenBiases = new double[layerCollection[i].Activations.Length];

                for (int j = 0; j < layerCollection[i - 1].Activations.Length; j++)
                {
                    inputBiases[i] = 0;
                }

                inputBiasesList.Add(inputBiases);

                for (int j = 0; j < layerCollection[i].Activations.Length; j++)
                {
                    hiddenBiases[i] = 0;
                }

                hiddenBiasesList.Add(hiddenBiases);
            }

            for (int i = 0; i < layerCollection.Count - 2; i++)
            {
                int t = 0;

                while (t < epochs)
                {
                    foreach (int[] vector in vectorList)
                    {
                        int[] inputVector = new int[vector.Length];
                        int j = 1;

                        for (int k = 0; k < vector.Length; k++)
                        {
                            inputVector[k] = vector[k];
                        }

                        while (j <= i)
                        {
                            int[] tempVector = new int[inputVector.Length];

                            for (int k = 0; k < inputVector.Length; k++)
                            {
                                tempVector[k] = inputVector[k];
                            }

                            inputVector = new int[layerCollection[j].Activations.Length];

                            double[] summations = new double[layerCollection[j].Activations.Length];

                            for (int k = 0; k < inputVector.Length; k++)
                            {
                                double sum = 0;

                                for (int l = 0; l < tempVector.Length; l++)
                                {
                                    sum += weightsCollection[j][k, l] * tempVector[l];
                                }

                                sum += inputBiasesList[j][k];
                                summations[k] = sum;
                            }

                            for (int k = 0; k < summations.Length; k++)
                            {
                                inputVector[k] = Binomial(1, layerCollection[j].ActivationFunction.Function(summations, k));
                            }

                            j++;
                        }

                        Layer yLayer = layerCollection[i + 1];
                        int[] x = new int[i == 0 ? vector.Length : layerCollection[i].Activations.Length];
                        double[] y = new double[yLayer.Activations.Length];
                        double[] z = new double[x.Length];
                        double[] ySummations = new double[y.Length];
                        double[] zSummations = new double[z.Length];
                        double[] tempInputBiases = new double[x.Length];
                        double[] tempHiddenBiases = new double[y.Length];

                        for (int n = 0; n < x.Length; n++)
                        {
                            if (inputVector[n] == 0)
                            {
                                x[n] = 0;
                            }
                            else
                            {
                                x[n] = Binomial(1, 1.0 - this.corruptionLevel);
                            }
                        }

                        // Encode
                        for (int n = 0; n < y.Length; n++)
                        {
                            double sum = 0;

                            for (int m = 0; m < x.Length; m++)
                            {
                                sum += weightsCollection[i][m, n] * x[m];
                            }

                            sum += hiddenBiasesList[i][n];
                            ySummations[n] = sum;
                        }

                        for (int n = 0; n < ySummations.Length; n++)
                        {
                            y[n] = yLayer.ActivationFunction.Function(ySummations, n);
                        }

                        // Decode
                        for (int n = 0; n < z.Length; n++)
                        {
                            double sum = 0;

                            for (int m = 0; m < y.Length; m++)
                            {
                                sum += weightsCollection[i][n, m] * y[m];
                            }

                            sum += inputBiasesList[i][n];
                            zSummations[n] = sum;
                        }

                        for (int n = 0; n < zSummations.Length; n++)
                        {
                            z[n] = yLayer.ActivationFunction.Function(zSummations, n);
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
                                tempHiddenBiases[n] += weightsCollection[i][m, n] * tempInputBiases[m];
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
                                    weightsCollection[i][n, m] += this.learningRate * (tempHiddenBiases[m] * x[n] + tempInputBiases[n] * y[m]) / vectorList.Count;
                                }
                                else
                                {
                                    weightsCollection[i][n, m] = inputBiasesList[i][n];
                                }
                            }
                        }
                    }

                    t++;
                }
            }
        }

        private int Binomial(int n, double p)
        {
            int count = 0;

            for (int i = 0; i < n; i++)
            {
                if (this.random.NextDouble() < p)
                {
                    count++;
                }
            }

            return count;
        }
    }
}
