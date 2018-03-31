using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Megalopolis.Layers;

namespace Megalopolis
{
    public class Network
    {
        private Random random = null;
        private Collection<Layer> layerCollection = null;
        private ITrainer trainer = null;

        public IEnumerable<Layer> Layers
        {
            get
            {
                return this.layerCollection;
            }
        }

        public Network(Random random, IEnumerable<Layer> layers, Func<int, int, double> minFunc, Func<int, int, double> maxFunc, ITrainer trainer)
        {
            Layer previousLayer = null;

            this.random = random;
            this.layerCollection = new Collection<Layer>();
            this.trainer = trainer;

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

        public Network(Random random, IEnumerable<Layer> layers, Func<int, double> weightFunc, Func<int, double> biasFunc, ITrainer trainer)
        {
            Layer previousLayer = null;
            int weightIndex = 0;
            int biasIndex = 0;

            this.random = random;
            this.layerCollection = new Collection<Layer>();
            this.trainer = trainer;

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
            this.trainer.Train(this.layerCollection, dictionary, epochs);
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
    }
}
