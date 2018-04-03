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
        private Layer inputLayer = null;
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

        public Network(Random random, Layer layer, IOptimizer optimizer, ILossFunction lossFunction)
        {
            this.random = random;
            this.inputLayer = layer;
            this.layerCollection = new Collection<Layer>();
            this.optimizer = optimizer;
            this.lossFunction = lossFunction;

            do
            {
                this.layerCollection.Add(layer);
                layer = layer.Next;
            } while (layer != null);
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
                    var batchOfGradients = new double[this.layerCollection.Count][];

                    foreach (var keyValuePair in keyValuePairList.Sample<KeyValuePair<double[], double[]>>(this.random, Math.Min(remaining, this.batchSize)))
                    {
                        int i = 0;

                        foreach (var gradients in BackwardPropagate(ForwardPropagate(true, this.inputLayer, keyValuePair.Key), keyValuePair.Value))
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

                this.loss = GetLoss(this.inputLayer, keyValuePairList);

                if (this.Stepped != null)
                {
                    this.Stepped(this, new EventArgs());
                }

                t++;
            }
        }

        public double[] Predicate(double[] vector)
        {
            var layer = this.inputLayer;
            Layer outputLayer;

            for (int i = 0; i < layer.InputActivations.Length; i++)
            {
                layer.InputActivations[i] = vector[i];
            }

            do
            {
                layer.PropagateForward(false);
                outputLayer = layer;
                layer = layer.Next;
            } while (layer != null);

            return outputLayer.OutputActivations;
        }

        private double GetLoss(Layer inputLayer, IEnumerable<KeyValuePair<double[], double[]>> keyValuePairs)
        {
            double sum = 0.0;

            foreach (var keyValuePair in keyValuePairs)
            {
                var layer = ForwardPropagate(false, inputLayer, keyValuePair.Key);

                for (int i = 0; i < layer.OutputActivations.Length; i++)
                {
                    sum += this.lossFunction.Function(layer.OutputActivations[i], keyValuePair.Value[i]);
                }
            }

            return sum;
        }

        private Layer ForwardPropagate(bool isTraining, Layer inputLayer, double[] vector)
        {
            var layer = inputLayer;
            Layer outputLayer;

            for (int i = 0; i < inputLayer.InputActivations.Length; i++)
            {
                inputLayer.InputActivations[i] = vector[i];
            }

            do
            {
                layer.PropagateForward(isTraining);
                outputLayer = layer;
                layer = layer.Next;
            } while (layer != null);

            return outputLayer;
        }

        private IEnumerable<double[]> BackwardPropagate(Layer outputLayer, double[] vector)
        {
            var layer = outputLayer.Previous;
            var gradientsList = new LinkedList<double[]>();
            var gradients = new double[outputLayer.OutputActivations.Length];

            for (int i = 0; i < outputLayer.OutputActivations.Length; i++)
            {
                gradients[i] = this.lossFunction.Derivative(outputLayer.OutputActivations[i], vector[i]);
            }

            foreach (var g in outputLayer.PropagateBackward(ref gradients))
            {
                gradientsList.AddFirst(g);
            }

            gradients = gradientsList.First.Value;

            while (layer != null)
            {
                var tempGradientsList = new LinkedList<double[]>();

                foreach (var g in layer.PropagateBackward(ref gradients))
                {
                    tempGradientsList.AddLast(g);
                }

                gradientsList.First.Value = gradients;
                gradients = tempGradientsList.Last.Value;

                foreach (var g in tempGradientsList)
                {
                    gradientsList.AddFirst(g);
                }

                layer = layer.Previous;
            }

            gradientsList.RemoveFirst();

            return gradientsList;
        }
    }
}
