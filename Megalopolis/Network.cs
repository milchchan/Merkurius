using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Megalopolis.Optimizers;
using Megalopolis.Layers;
using Megalopolis.LossFunctions;

namespace Megalopolis
{
    public class Network
    {
        public event EventHandler<EventArgs> Stepped = null;
        private Random random = null;
        private Collection<Layer> layerCollection = null;
        private int batchSize = 32;
        private double loss = 0;
        private IOptimizer optimizer = null;
        private ILossFunction lossFunction = null;

        public IEnumerable<Layer> Layers
        {
            get
            {
                return this.layerCollection;
            }
        }

        public int BatchSize
        {
            get
            {
                return this.batchSize;
            }
        }

        public double Loss
        {
            get
            {
                return this.loss;
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

        public Network(Random random, IEnumerable<Layer> layers, Func<int, int, double> minFunc, Func<int, int, double> maxFunc, IOptimizer optimizer, ILossFunction lossFunction)
        {
            Layer previousLayer = null;

            this.random = random;
            this.layerCollection = new Collection<Layer>();
            this.optimizer = optimizer;
            this.lossFunction = lossFunction;

            foreach (var layer in layers)
            {
                if (previousLayer != null)
                {
                    double min = minFunc(previousLayer.Activations.Length, layer.Activations.Length);
                    double max = maxFunc(previousLayer.Activations.Length, layer.Activations.Length);

                    previousLayer.Connect(layer);

                    for (int i = 0; i < previousLayer.Activations.Length; i++)
                    {
                        for (int j = 0; j < layer.Activations.Length; j++)
                        {
                            previousLayer.Weights[i, j] = random.Uniform(min, max);
                        }
                    }

                    for (int i = 0; i < layer.Activations.Length; i++)
                    {
                        previousLayer.Biases[i] = random.Uniform(min, max);
                    }
                }

                previousLayer = layer;
                this.layerCollection.Add(layer);
            }
        }

        public Network(Random random, IEnumerable<Layer> layers, Func<int, double> weightFunc, Func<int, double> biasFunc, IOptimizer optimizer, ILossFunction lossFunction)
        {
            Layer previousLayer = null;
            int weightIndex = 0;
            int biasIndex = 0;

            this.random = random;
            this.layerCollection = new Collection<Layer>();
            this.optimizer = optimizer;
            this.lossFunction = lossFunction;

            foreach (var layer in layers)
            {
                if (previousLayer != null)
                {
                    previousLayer.Connect(layer);

                    for (int i = 0; i < previousLayer.Activations.Length; i++)
                    {
                        for (int j = 0; j < layer.Activations.Length; j++)
                        {
                            previousLayer.Weights[i, j] = weightFunc(weightIndex);
                            weightIndex++;
                        }
                    }

                    for (int i = 0; i < layer.Activations.Length; i++)
                    {
                        previousLayer.Biases[i] = biasFunc(biasIndex);
                        biasIndex++;
                    }
                }

                previousLayer = layer;
                this.layerCollection.Add(layer);
            }
        }

        public void Train(IDictionary<double[], IEnumerable<double[]>> dictionary, int epochs)
        {
            // Backpropagation
            List<KeyValuePair<double[], double[]>> keyValuePairList = dictionary.Aggregate<KeyValuePair<double[], IEnumerable<double[]>>, List<KeyValuePair<double[], double[]>>>(new List<KeyValuePair<double[], double[]>>(), (list, kvp) =>
            {
                foreach (var vector in kvp.Value)
                {
                    list.Add(new KeyValuePair<double[], double[]>(vector, kvp.Key));
                }

                return list;
            });
            int t = 0;

            // Stochastic gradient descent (SGD)
            while (t < epochs)
            {
                // Mini-batch
                int remaining = keyValuePairList.Count;

                do
                {
                    var batchOfGradients = new double[this.layerCollection.Count - 1][];

                    foreach (var keyValuePair in keyValuePairList.Sample<KeyValuePair<double[], double[]>>(this.random, Math.Min(remaining, this.batchSize)))
                    {
                        int i = 0;

                        foreach (var gradients in BackwardPropagate(ForwardPropagate(true, this.layerCollection[0], keyValuePair.Key), keyValuePair.Value))
                        {
                            batchOfGradients[i] = gradients;
                            i++;
                        }
                    }

                    for (int i = 0, j = 0; i < batchOfGradients.Length; i++)
                    {
                        var gradients = new double[batchOfGradients[i].Length];

                        for (int k = 0; k < batchOfGradients[i].Length; k++)
                        {
                            gradients[k] = batchOfGradients[i][k] / batchOfGradients.Length;
                        }

                        this.layerCollection[i].Update(gradients, (weight, gradient) => optimizer.Optimize(j++, weight, gradient));
                    }

                    remaining -= this.batchSize;
                } while (remaining > 0);

                this.loss = GetLoss(this.layerCollection[0], keyValuePairList);

                if (this.Stepped != null)
                {
                    this.Stepped(this, new EventArgs());
                }

                t++;
            }
        }

        public double[] Predicate(double[] vector)
        {
            var layer = this.layerCollection[0];

            for (int i = 0; i < layer.Activations.Length; i++)
            {
                layer.Activations[i] = vector[i];
            }

            do
            {
                layer.PropagateForward(false);
                layer = layer.Next;
            } while (layer.Next != null);

            return layer.Activations;
        }

        private double GetLoss(Layer inputLayer, IEnumerable<KeyValuePair<double[], double[]>> keyValuePairs)
        {
            double sum = 0.0;

            foreach (var keyValuePair in keyValuePairs)
            {
                var layer = ForwardPropagate(false, inputLayer, keyValuePair.Key);

                for (int i = 0; i < layer.Activations.Length; i++)
                {
                    sum += this.lossFunction.Function(layer.Activations[i], keyValuePair.Value[i]);
                }
            }

            return sum;
        }

        private Layer ForwardPropagate(bool isTraining, Layer inputLayer, double[] vector)
        {
            var layer = inputLayer;

            for (int i = 0; i < inputLayer.Activations.Length; i++)
            {
                inputLayer.Activations[i] = vector[i];
            }

            do
            {
                layer.PropagateForward(isTraining);
                layer = layer.Next;
            } while (layer.Next != null);

            return layer;
        }

        private IEnumerable<double[]> BackwardPropagate(Layer outputLayer, double[] vector)
        {
            var layer = outputLayer.Previous;
            var gradientsList = new LinkedList<double[]>();
            var gradients = new double[outputLayer.Activations.Length];

            for (int i = 0; i < outputLayer.Activations.Length; i++)
            {
                gradients[i] = this.lossFunction.Derivative(outputLayer.Activations[i], vector[i]);
            }

            gradients = outputLayer.PropagateBackward(ref gradients);
            gradientsList.AddFirst(gradients);

            do
            {
                var tempGradients = layer.PropagateBackward(ref gradients);

                gradientsList.First.Value = gradients;
                gradients = tempGradients;
                gradientsList.AddFirst(gradients);
                layer = layer.Previous;
            } while (layer.Previous != null);

            return gradientsList;
        }
    }
}
