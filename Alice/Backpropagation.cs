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
        private double loss = 0;
        private IOptimizer optimizer = null;
        private ILossFunction lossFunction = null;

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

            // Stochastic gradient descent
            while (t < epochs)
            {
                foreach (var keyValuePair in keyValuePairList.Shuffle<KeyValuePair<double[], double[]>>(this.random))
                {
                    int identifier = 0;
                    int index = 0;

                    foreach (var gradients in BackwardPropagate(ForwardPropagate(inputLayer, keyValuePair.Key), keyValuePair.Value))
                    {
                        var layer = layerCollection[index];

                        for (int i = 0; i < layer.Activations.Length; i++)
                        {
                            for (int j = 0; j < layer.Next.Activations.Length; j++)
                            {
                                layer.Weights[i, j] = optimizer.Optimize(identifier, layer.Weights[i, j], gradients[j] * layer.Activations[i]);
                                identifier++;
                            }
                        }

                        for (int i = 0; i < layer.Next.Activations.Length; i++)
                        {
                            layer.Biases[i] = optimizer.Optimize(identifier, layer.Biases[i], gradients[i]);
                            identifier++;
                        }

                        index++;
                    }
                }

                t++;
            }

            this.loss = GetLoss(inputLayer, keyValuePairList);
        }

        private double GetLoss(Layer inputLayer, IEnumerable<KeyValuePair<double[], double[]>> keyValuePairs)
        {
            double sum = 0.0;

            foreach (var keyValuePair in keyValuePairs)
            {
                var layer = ForwardPropagate(inputLayer, keyValuePair.Key);

                for (int i = 0; i < layer.Activations.Length; i++)
                {
                    sum += this.lossFunction.Function(layer.Activations[i], keyValuePair.Value[i]);
                }
            }

            return sum;
        }

        private Layer ForwardPropagate(Layer inputLayer, double[] vector)
        {
            var layer = inputLayer;

            for (int i = 0; i < inputLayer.Activations.Length; i++)
            {
                inputLayer.Activations[i] = vector[i];
            }

            do
            {
                layer.PropagateForward();
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
