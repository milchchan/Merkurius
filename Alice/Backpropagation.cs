using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Alice.Optimizers;
using Alice.Layers;
using Alice.LossFunctions;

namespace Alice
{
    public class Backpropagation : ITrainer
    {
        private Random random = null;
        private int batchSize = 32;
        private double loss = 0;
        private IOptimizer optimizer = null;
        private ILossFunction lossFunction = null;

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

        public Backpropagation(Random random, IOptimizer optimizer, ILossFunction lossFunction)
        {
            this.random = random;
            this.optimizer = optimizer;
            this.lossFunction = lossFunction;
        }

        public void Train(Collection<Layer> layerCollection, IDictionary<double[], IEnumerable<double[]>> dictionary, int epochs)
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
            var inputLayer = layerCollection[0];

            // Stochastic gradient descent (SGD)
            while (t < epochs)
            {
                // Mini-batch
                int remaining = keyValuePairList.Count;

                do
                {
                    var batchOfGradients = new double[layerCollection.Count - 1][];
                    int index = 0;

                    foreach (var keyValuePair in keyValuePairList.Sample<KeyValuePair<double[], double[]>>(this.random, Math.Min(remaining, this.batchSize)))
                    {
                        int i = 0;

                        foreach (var gradients in BackwardPropagate(ForwardPropagate(true, inputLayer, keyValuePair.Key), keyValuePair.Value))
                        {
                            batchOfGradients[i] = gradients;
                            i++;
                        }
                    }

                    for (int i = 0; i < batchOfGradients.Length; i++)
                    {
                        for (int j = 0; j < layerCollection[i].Activations.Length; j++)
                        {
                            for (int k = 0; k < layerCollection[i].Next.Activations.Length; k++)
                            {
                                layerCollection[i].Weights[j, k] = optimizer.Optimize(index, layerCollection[i].Weights[j, k], batchOfGradients[i][k] * layerCollection[i].Activations[j] / batchOfGradients.Length);
                                index++;
                            }
                        }

                        for (int j = 0; j < layerCollection[i].Next.Activations.Length; j++)
                        {
                            layerCollection[i].Biases[j] = optimizer.Optimize(index, layerCollection[i].Biases[j], batchOfGradients[i][j] / batchOfGradients.Length);
                            index++;
                        }
                    }

                    remaining -= this.batchSize;
                } while (remaining > 0);
                
                t++;
            }

            this.loss = GetLoss(inputLayer, keyValuePairList);
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
            var layer = outputLayer;
            var gradientsList = new LinkedList<double[]>();
            var gradients = new double[outputLayer.Activations.Length];

            for (int i = 0; i < outputLayer.Activations.Length; i++)
            {
                gradients[i] = this.lossFunction.Derivative(outputLayer.Activations[i], vector[i]);
            }

            do
            {
                gradients = layer.PropagateBackward(gradients);
                gradientsList.AddFirst(gradients);
                layer = layer.Previous;
            } while (layer.Previous != null);

            return gradientsList;
        }
    }
}
