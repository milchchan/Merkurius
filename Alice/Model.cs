using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace Alice
{
    public class Model
    {
        private Collection<Layer> layerCollection = null;
        private Collection<double[,]> weightsCollection = null;
        private ITrainer trainer = null;

        public IEnumerable<Layer> Layers
        {
            get
            {
                return this.layerCollection;
            }
        }

        public IEnumerable<double[,]> Weights
        {
            get
            {
                return this.weightsCollection;
            }
        }

        public Model(int seed, int inputNodes, int hiddenNodes, int hiddenLayers, int outputNodes, Func<int, int, double> minFunc, Func<int, int, double> maxFunc, Alice.ActivationFunctions.IActivationFunction activationFunction, Alice.Optimizers.IOptimizer optimizer)
        {
            Random random = new Random(seed);
            int layers = 2 + hiddenLayers;

            this.layerCollection = new Collection<Layer>();
            this.weightsCollection = new Collection<double[,]>();
            this.trainer = new Backpropagation(random, new Alice.Optimizers.AdaDelta(), new Alice.LossFunctions.MeanSquaredError());

            for (int i = 0; i < layers; i++)
            {
                if (i == 0)
                {
                    this.layerCollection.Add(new Layer(inputNodes + 1, activationFunction)); // + 1 for bias node
                }
                else
                {
                    int nodes = i == layers - 1 ? outputNodes : hiddenNodes;

                    this.layerCollection.Add(new Layer(nodes, activationFunction));

                    double[,] weights = new double[this.layerCollection[i - 1].Activations.Length, nodes];
                    double min = minFunc(this.layerCollection[i - 1].Activations.Length, nodes);
                    double max = maxFunc(this.layerCollection[i - 1].Activations.Length, nodes);

                    for (int j = 0; j < this.layerCollection[i - 1].Activations.Length; j++)
                    {
                        for (int k = 0; k < nodes; k++)
                        {
                            weights[j, k] = Uniform(random, min, max);
                        }
                    }

                    this.weightsCollection.Add(weights);
                }
            }
        }

        public Model(IEnumerable<Layer> layers, ITrainer trainer)
        {
            this.layerCollection = new Collection<Layer>();
            this.weightsCollection = new Collection<double[,]>();
            this.trainer = trainer;

            foreach (Layer layer in layers)
            {
                if (this.layerCollection.Count > 0)
                {
                    int nodes = this.layerCollection[this.layerCollection.Count - 1].Activations.Length;
                    double[,] weights = new double[nodes, layer.Activations.Length];

                    for (int i = 0; i < nodes; i++)
                    {
                        for (int j = 0; j < layer.Activations.Length; j++)
                        {
                            weights[i, j] = 0;
                        }
                    }

                    this.weightsCollection.Add(weights);
                }

                this.layerCollection.Add(layer);
            }
        }

        public Model(IEnumerable<Layer> layers, Func<int, double> func, ITrainer trainer)
        {
            int index = 0;

            this.layerCollection = new Collection<Layer>();
            this.weightsCollection = new Collection<double[,]>();
            this.trainer = trainer;

            foreach (Layer layer in layers)
            {
                if (this.layerCollection.Count > 0)
                {
                    int nodes = this.layerCollection[this.layerCollection.Count - 1].Activations.Length;
                    double[,] weights = new double[nodes, layer.Activations.Length];

                    for (int i = 0; i < nodes; i++)
                    {
                        for (int j = 0; j < layer.Activations.Length; j++)
                        {
                            weights[i, j] = func(index);
                            index++;
                        }
                    }

                    this.weightsCollection.Add(weights);
                }

                this.layerCollection.Add(layer);
            }
        }

        public Model(Random random, IEnumerable<Layer> layers, Func<int, int, double> minFunc, Func<int, int, double> maxFunc, ITrainer trainer)
        {
            this.layerCollection = new Collection<Layer>();
            this.weightsCollection = new Collection<double[,]>();
            this.trainer = trainer;

            foreach (Layer layer in layers)
            {
                if (this.layerCollection.Count > 0)
                {
                    int nodes = this.layerCollection[this.layerCollection.Count - 1].Activations.Length;
                    double[,] weights = new double[nodes, layer.Activations.Length];
                    double min = minFunc(nodes, layer.Activations.Length);
                    double max = maxFunc(nodes, layer.Activations.Length);

                    for (int i = 0; i < nodes; i++)
                    {
                        for (int j = 0; j < layer.Activations.Length; j++)
                        {
                            weights[i, j] = Uniform(random, min, max);
                        }
                    }

                    this.weightsCollection.Add(weights);
                }

                this.layerCollection.Add(layer);
            }
        }

        public void Train(IDictionary<double[], IEnumerable<double[]>> dictionary, int epochs)
        {
            this.trainer.Train(this.layerCollection, this.weightsCollection, dictionary, epochs);
        }

        public double[] Predicate(double[] vector)
        {
            double[][] tempActivations = new double[this.layerCollection.Count][];

            for (int i = 0; i < this.layerCollection.Count; i++)
            {
                tempActivations[i] = new double[this.layerCollection[i].Activations.Length];

                for (int j = 0; j < this.layerCollection[i].Activations.Length; j++)
                {
                    tempActivations[i][j] = this.layerCollection[i].Activations[j];
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
                    double[] summations = new double[tempActivations[i].Length];

                    for (int j = 0; j < tempActivations[i].Length; j++)
                    {
                        double sum = 0;

                        for (int k = 0; k < tempActivations[i - 1].Length; k++)
                        {
                            sum += tempActivations[i - 1][k] * this.weightsCollection[i - 1][k, j];
                        }

                        summations[j] = sum;
                    }

                    for (int j = 0; j < tempActivations[i].Length; j++)
                    {
                        tempActivations[i][j] = this.layerCollection[i].ActivationFunction.Function(summations, j);
                    }
                }
            }

            return tempActivations[tempActivations.Length - 1];
        }

        private double Uniform(Random random, double min, double max)
        {
            return (max - min) * random.NextDouble() + min;
        }
    }
}
