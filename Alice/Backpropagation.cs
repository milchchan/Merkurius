using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Alice.Optimizers;
using Alice.LossFunctions;

namespace Alice
{
    public class Backpropagation : ITrainer
    {
        private Random random = null;
        private double errorThreshold = 0.01;
        private IOptimizer optimizer = null;
        private ILossFunction lossFunction = null;

        public double ErrorThreshold
        {
            get
            {
                return this.errorThreshold;
            }
            set
            {
                this.errorThreshold = value;
            }
        }

        public IOptimizer Optimizer
        {
            get
            {
                return this.optimizer;
            }
        }

        public ILossFunction LossFunction
        {
            get
            {
                return this.lossFunction;
            }
        }

        public Backpropagation(Random random, IOptimizer optimizer, ILossFunction lossFunction)
        {
            this.random = random;
            this.optimizer = optimizer;
            this.lossFunction = lossFunction;
        }

        public void Train(Collection<Layer> layerCollection, Collection<double[,]> weightsCollection, IDictionary<double[], IEnumerable<double[]>> dictionary, int epochs)
        {
            // Backpropagation
            List<KeyValuePair<double[], double[]>> kvpList = dictionary.Aggregate<KeyValuePair<double[], IEnumerable<double[]>>, List<KeyValuePair<double[], double[]>>>(new List<KeyValuePair<double[], double[]>>(), (list, kvp) =>
            {
                foreach (double[] vector in kvp.Value)
                {
                    list.Add(new KeyValuePair<double[], double[]>(vector, kvp.Key));
                }

                return list;
            });
            int t = 0;
            int hiddenLayers = layerCollection.Count - 2;
            Layer outputLayer = layerCollection[layerCollection.Count - 1];

            // Stochastic gradient descent
            while (t < epochs)
            {
                double error = 0;

                foreach (KeyValuePair<double[], double[]> kvp in Shuffle<KeyValuePair<double[], double[]>>(kvpList))
                {
                    List<int[]> dropoutList;
                    double[][] deltas;
                    int index = 0;

                    ForwardPropagate(layerCollection, weightsCollection, kvp.Key, out dropoutList);

                    BackwardPropagate(layerCollection, weightsCollection, kvp.Value, dropoutList, out deltas);

                    for (int i = hiddenLayers; i >= 0; i--)
                    {
                        Layer layer1 = layerCollection[i];
                        Layer layer2 = layerCollection[i + 1];

                        for (int j = 0; j < layer1.Activations.Length; j++)
                        {
                            for (int k = 0; k < layer2.Activations.Length; k++)
                            {
                                weightsCollection[i][j, k] = optimizer.Optimize(index, weightsCollection[i][j, k], deltas[i][k] * layer1.Activations[j]);
                                index++;
                            }
                        }
                    }

                    for (int i = 0; i < kvp.Value.Length; i++)
                    {
                        error += this.lossFunction.Function(kvp.Value[i], outputLayer.Activations[i]);
                    }
                }

                if (error < this.errorThreshold)
                {
                    break;
                }

                t++;
            }
        }

        private void ForwardPropagate(Collection<Layer> layerCollection, Collection<double[,]> weightsCollection, double[] vector, out List<int[]> dropoutList)
        {
            dropoutList = new List<int[]>();

            for (int i = 0; i < layerCollection.Count; i++)
            {
                int[] mask = new int[layerCollection[i].Activations.Length];

                if (i == 0)
                {
                    for (int j = 0; j < layerCollection[i].Activations.Length - 1; j++)
                    {
                        mask[j] = Binomial(1, layerCollection[i].DropoutProbability);
                        layerCollection[i].Activations[j] = vector[j] * mask[j];
                    }
                }
                else
                {
                    double[] summations = new double[layerCollection[i].Activations.Length];

                    for (int j = 0; j < layerCollection[i].Activations.Length; j++)
                    {
                        double sum = 0;

                        for (int k = 0; k < layerCollection[i - 1].Activations.Length; k++)
                        {
                            sum += layerCollection[i - 1].Activations[k] * weightsCollection[i - 1][k, j];
                        }

                        summations[j] = sum;
                    }

                    for (int j = 0; j < layerCollection[i].Activations.Length; j++)
                    {
                        mask[j] = Binomial(1, layerCollection[i].DropoutProbability);
                        layerCollection[i].Activations[j] = layerCollection[i].ActivationFunction.Function(summations, j) * mask[j];
                    }
                }

                dropoutList.Add(mask);
            }
        }

        private void BackwardPropagate(Collection<Layer> layerCollection, Collection<double[,]> weightsCollection, double[] vector, List<int[]> dropoutList, out double[][] deltas)
        {
            Layer outputLayer = layerCollection[layerCollection.Count - 1];
            int index = layerCollection.Count - 2;

            deltas = new double[layerCollection.Count - 1][];
            deltas[index] = new double[outputLayer.Activations.Length];

            for (int i = 0; i < outputLayer.Activations.Length; i++)
            {
                deltas[index][i] = layerCollection[index].ActivationFunction.Derivative(outputLayer.Activations, i) * this.lossFunction.Derivative(outputLayer.Activations[i], vector[i]) * dropoutList[layerCollection.Count - 1][i];
            }

            for (int i = layerCollection.Count - 2; i > 0; i--)
            {
                int previousIndex = i - 1;
                int nextIndex = i + 1;

                deltas[previousIndex] = new double[layerCollection[i].Activations.Length];

                for (int j = 0; j < layerCollection[i].Activations.Length; j++)
                {
                    double error = 0;

                    for (int k = 0; k < layerCollection[nextIndex].Activations.Length; k++)
                    {
                        error += deltas[i][k] * weightsCollection[i][j, k];
                    }

                    deltas[previousIndex][j] = layerCollection[previousIndex].ActivationFunction.Derivative(layerCollection[i].Activations, j) * error * dropoutList[i][j];
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

        private IEnumerable<T> Shuffle<T>(IEnumerable<T> collection)
        {
            // Fisher-Yates algorithm
            T[] array = collection.ToArray();
            int n = array.Length; // The number of items left to shuffle (loop invariant).

            while (n > 1)
            {
                int k = this.random.Next(n); // 0 <= k < n.

                n--; // n is now the last pertinent index;
                T temp = array[n]; // swap list[n] with list[k] (does nothing if k == n).
                array[n] = array[k];
                array[k] = temp;
            }

            return array;
        }
    }
}
