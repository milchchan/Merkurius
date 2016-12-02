using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Milk.ActivationFunctions;
using Milk.LossFunctions;
using Milk.Optimizers;

namespace Milk
{
    public class Model
    {
        private Random random = null;
        private List<Layer> layerList = null;
        private Collection<double[,]> weightsCollection = null;
        private double errorThreshold = 0.01;
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

        public IEnumerable<Layer> Layers
        {
            get
            {
                return this.layerList;
            }
        }

        public Collection<double[,]> Weights
        {
            get
            {
                return this.weightsCollection;
            }
            set
            {
                this.weightsCollection = value;
            }
        }

        public Model(int seed, int inputNodes, int hiddenNodes, int hiddenLayers, int outputNodes, Func<int, int, double> minFunc, Func<int, int, double> maxFunc, IActivationFunction activationFunction, IOptimizer optimizer)
        {
            int layers = 2 + hiddenLayers;

            this.random = new Random(seed);
            this.layerList = new List<Layer>();
            this.weightsCollection = new Collection<double[,]>();
            this.lossFunction = new MeanSquaredError();

            for (int i = 0; i < layers; i++)
            {
                if (i == 0)
                {
                    this.layerList.Add(new Layer(inputNodes + 1, activationFunction, optimizer)); // + 1 for bias node
                }
                else
                {
                    int nodes = i == layers - 1 ? outputNodes : hiddenNodes;

                    this.layerList.Add(new Layer(nodes, activationFunction, optimizer));

                    double[,] weights = new double[this.layerList[i - 1].Activations.Length, nodes];
                    double min = minFunc(this.layerList[i - 1].Activations.Length, nodes);
                    double max = maxFunc(this.layerList[i - 1].Activations.Length, nodes);

                    for (int j = 0; j < this.layerList[i - 1].Activations.Length; j++)
                    {
                        for (int k = 0; k < nodes; k++)
                        {
                            weights[j, k] = Uniform(min, max);
                        }
                    }

                    this.weightsCollection.Add(weights);
                }
            }
        }

        public Model(Random random, IEnumerable<Layer> layers, Func<int, int, double> minFunc, Func<int, int, double> maxFunc, ILossFunction lossFunction)
        {
            this.random = random;
            this.layerList = new List<Layer>();
            this.weightsCollection = new Collection<double[,]>();
            this.lossFunction = lossFunction;

            foreach (Layer layer in layers)
            {
                if (this.layerList.Count > 0)
                {
                    int nodes = this.layerList[this.layerList.Count - 1].Activations.Length;
                    double[,] weights = new double[nodes, layer.Activations.Length];
                    double min = minFunc(nodes, layer.Activations.Length);
                    double max = maxFunc(nodes, layer.Activations.Length);

                    for (int i = 0; i < nodes; i++)
                    {
                        for (int j = 0; j < layer.Activations.Length; j++)
                        {
                            weights[i, j] = Uniform(min, max);
                        }
                    }

                    this.weightsCollection.Add(weights);
                }

                this.layerList.Add(layer);
            }
        }

        public void Pretrain(IEnumerable<double[]> inputs, double learningRate, double corruptionLevel, int epochs)
        {
            // Stacked Denoising Autoencoders (SdA)
            int numberOfInputs = inputs.Count();
            List<double[]> inputBiasesList = new List<double[]>();
            List<double[]> hiddenBiasesList = new List<double[]>();
            
            for (int i = 1; i < this.layerList.Count - 1; i++)
            {
                double[] inputBiases = new double[this.layerList[i - 1].Activations.Length];
                double[] hiddenBiases = new double[this.layerList[i].Activations.Length];

                for (int j = 0; j < this.layerList[i - 1].Activations.Length; j++)
                {
                    inputBiases[i] = 0;
                }

                inputBiasesList.Add(inputBiases);

                for (int j = 0; j < this.layerList[i].Activations.Length; j++)
                {
                    hiddenBiases[i] = 0;
                }

                hiddenBiasesList.Add(hiddenBiases);
            }

            for (int i = 0; i < this.layerList.Count - 2; i++)
            {
                int t = 0;

                while (t < epochs)
                {
                    foreach (double[] vector in inputs)
                    {
                        int[] inputVector = new int[vector.Length];
                        int j = 1;

                        for (int k = 0; k < vector.Length; k++)
                        {
                            inputVector[k] = (int)vector[k];
                        }

                        while (j <= i)
                        {
                            int[] tempVector = new int[inputVector.Length];

                            for (int k = 0; k < inputVector.Length; k++)
                            {
                                tempVector[k] = inputVector[k];
                            }

                            inputVector = new int[this.layerList[j].Activations.Length];

                            for (int k = 0; k < inputVector.Length; k++)
                            {
                                double sum = 0;

                                for (int l = 0; l < tempVector.Length; l++)
                                {
                                    sum += this.weightsCollection[j][k, l] * tempVector[l];
                                }

                                sum += inputBiasesList[j][k];
                                inputVector[k] = Binomial(1, this.layerList[j].ActivationFunction.Function(sum));
                            }

                            j++;
                        }

                        int[] x = new int[i == 0 ? vector.Length : this.layerList[i].Activations.Length];
                        double[] y = new double[this.layerList[i + 1].Activations.Length];
                        double[] z = new double[x.Length];
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
                                x[n] = Binomial(1, 1.0 - corruptionLevel);
                            }
                        }

                        // Encode
                        for (int n = 0; n < y.Length; n++)
                        {
                            y[n] = 0;

                            for (int m = 0; m < x.Length; m++)
                            {
                                y[n] += this.weightsCollection[i][m, n] * x[m];
                            }

                            y[n] += hiddenBiasesList[i][n];
                            y[n] = this.layerList[i + 1].ActivationFunction.Function(y[n]);
                        }

                        // Decode
                        for (int n = 0; n < z.Length; n++)
                        {
                            z[n] = 0;

                            for (int m = 0; m < y.Length; m++)
                            {
                                z[n] += this.weightsCollection[i][n, m] * y[m];
                            }

                            z[n] += inputBiasesList[i][n];
                            z[n] = this.layerList[i + 1].ActivationFunction.Function(z[n]);
                        }

                        for (int n = 0; n < z.Length; n++)
                        {
                            tempInputBiases[n] = inputVector[n] - z[n];
                            inputBiasesList[i][n] += learningRate * tempInputBiases[n] / numberOfInputs;
                        }

                        for (int n = 0; n < tempHiddenBiases.Length; n++)
                        {
                            tempHiddenBiases[n] = 0;

                            for (int m = 0; m < tempInputBiases.Length; m++)
                            {
                                tempHiddenBiases[n] += this.weightsCollection[i][m, n] * tempInputBiases[m];
                            }

                            tempHiddenBiases[n] *= y[n] * (1 - y[n]);
                            hiddenBiasesList[i][n] += learningRate * tempHiddenBiases[n] / numberOfInputs;
                        }

                        for (int n = 0; n < this.layerList[i].Activations.Length; n++)
                        {
                            for (int m = 0; m < this.layerList[i + 1].Activations.Length; m++)
                            {
                                if (n < x.Length)
                                {
                                    this.weightsCollection[i][n, m] += learningRate * (tempHiddenBiases[m] * x[n] + tempInputBiases[n] * y[m]) / numberOfInputs;
                                }
                                else
                                {
                                    this.weightsCollection[i][n, m] = inputBiasesList[i][n];
                                }
                            }
                        }
                    }

                    t++;
                }
            }
        }

        public void Train(IDictionary<double[], IEnumerable<double[]>> dictionary, int epochs)
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

            // Stochastic gradient descent
            while (t < epochs)
            {
                double error = 0;

                foreach (KeyValuePair<double[], double[]> kvp in Shuffle<KeyValuePair<double[], double[]>>(kvpList))
                {
                    List<int[]> dropoutList;
                    double[][] deltas;
                    int index = 0;

                    ForwardPropagate(kvp.Key, out dropoutList);

                    BackwardPropagate(kvp.Value, dropoutList, out deltas);

                    for (int i = this.layerList.Count - 2; i >= 0; i--)
                    {
                        for (int j = 0; j < this.layerList[i].Activations.Length; j++)
                        {
                            for (int k = 0; k < this.layerList[i + 1].Activations.Length; k++)
                            {
                                this.weightsCollection[i][j, k] = this.layerList[i].Optimizer.Optimize(index, this.weightsCollection[i][j, k], deltas[i][k] * this.layerList[i].Activations[j]);

                                index++;
                            }
                        }
                    }

                    for (int i = 0; i < kvp.Value.Length; i++)
                    {
                        error += this.lossFunction.Function(kvp.Value[i], this.layerList[this.layerList.Count - 1].Activations[i]);
                    }
                }

                if (error < this.errorThreshold)
                {
                    break;
                }

                t++;
            }
        }

        public double[] Predicate(double[] vector)
        {
            double[][] tempActivations = new double[this.layerList.Count][];

            for (int i = 0; i < this.layerList.Count; i++)
            {
                tempActivations[i] = new double[this.layerList[i].Activations.Length];

                for (int j = 0; j < this.layerList[i].Activations.Length; j++)
                {
                    tempActivations[i][j] = this.layerList[i].Activations[j];
                }
            }

            for (int i = 0; i < tempActivations.Length; i++)
            {
                if (i == 0)
                {
                    for (int j = 0; j < tempActivations[i].Length - 1; j++)
                    {
                        tempActivations[i][j] = vector[j];
                    }
                }
                else
                {
                    for (int j = 0; j < tempActivations[i].Length; j++)
                    {
                        double sum = 0;

                        for (int k = 0; k < tempActivations[i - 1].Length; k++)
                        {
                            sum += tempActivations[i - 1][k] * this.weightsCollection[i - 1][k, j];
                        }

                        tempActivations[i][j] = this.layerList[i].ActivationFunction.Function(sum);
                    }
                }
            }

            return tempActivations[tempActivations.Length - 1];
        }

        private void ForwardPropagate(double[] vector, out List<int[]> dropoutList)
        {
            dropoutList = new List<int[]>();

            for (int i = 0; i < this.layerList.Count; i++)
            {
                int[] mask = new int[this.layerList[i].Activations.Length];

                if (i == 0)
                {
                    for (int j = 0; j < this.layerList[i].Activations.Length - 1; j++)
                    {
                        mask[j] = Binomial(1, this.layerList[i].DropoutProbability);

                        this.layerList[i].Activations[j] = vector[j] * mask[j];
                    }
                }
                else
                {
                    for (int j = 0; j < this.layerList[i].Activations.Length; j++)
                    {
                        double sum = 0;

                        mask[j] = Binomial(1, this.layerList[i].DropoutProbability);

                        for (int k = 0; k < this.layerList[i - 1].Activations.Length; k++)
                        {
                            sum += this.layerList[i - 1].Activations[k] * this.weightsCollection[i - 1][k, j];
                        }

                        this.layerList[i].Activations[j] = this.layerList[i].ActivationFunction.Function(sum) * mask[j];
                    }
                }

                dropoutList.Add(mask);
            }
        }

        private void BackwardPropagate(double[] vector, List<int[]> dropoutList, out double[][] deltas)
        {
            Layer outputLayer = this.layerList[this.layerList.Count - 1];
            int index = this.layerList.Count - 2;

            deltas = new double[this.layerList.Count - 1][];
            deltas[index] = new double[outputLayer.Activations.Length];

            for (int i = 0; i < outputLayer.Activations.Length; i++)
            {
                deltas[index][i] = this.layerList[index].ActivationFunction.Derivative(outputLayer.Activations[i]) * this.lossFunction.Derivative(outputLayer.Activations[i], vector[i]) * dropoutList[this.layerList.Count - 1][i];
            }

            for (int i = this.layerList.Count - 2; i > 0; i--)
            {
                int previousIndex = i - 1;
                int nextIndex = i + 1;
                    
                deltas[previousIndex] = new double[this.layerList[i].Activations.Length];

                for (int j = 0; j < this.layerList[i].Activations.Length; j++)
                {
                    double error = 0;

                    for (int k = 0; k < this.layerList[nextIndex].Activations.Length; k++)
                    {
                        error += deltas[i][k] * this.weightsCollection[i][j, k];
                    }

                    deltas[previousIndex][j] = this.layerList[previousIndex].ActivationFunction.Derivative(this.layerList[i].Activations[j]) * error * dropoutList[i][j];
                }
            }
        }

        private double Uniform(double min, double max)
        {
            return (max - min) * this.random.NextDouble() + min;
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
