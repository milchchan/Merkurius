using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Megalopolis.Optimizers;
using Megalopolis.Layers;
using Megalopolis.LossFunctions;

namespace Megalopolis
{
    public class Model
    {
        public event EventHandler<EventArgs> Stepped = null;
        private Random random = null;
        private Collection<Layer> layerCollection = null;
        private double loss = 0.0;
        private IOptimizer optimizer = null;
        private ILossFunction lossFunction = null;
        private double weightDecayRate = 0.0;
        private double? maxGradient = null;

        public IEnumerable<Layer> Layers
        {
            get
            {
                return this.layerCollection;
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

        public double WeightDecayRate
        {
            get
            {
                return this.weightDecayRate;
            }
            set
            {
                this.weightDecayRate = value;
            }
        }

        public double? MaxGradient
        {
            get
            {
                return this.maxGradient;
            }
            set
            {
                this.maxGradient = value;
            }
        }

        public Model(Layer inputLayer, IOptimizer optimizer, ILossFunction lossFunction)
        {
            var layer = inputLayer;

            this.random = RandomProvider.GetRandom();
            this.layerCollection = new Collection<Layer>();
            this.optimizer = optimizer;
            this.lossFunction = lossFunction;

            do
            {
                this.layerCollection.Add(layer);
                layer = layer.Next;
            } while (layer != null);
        }

        public Model(IEnumerable<Layer> collection, IOptimizer optimizer, ILossFunction lossFunction)
        {
            this.random = RandomProvider.GetRandom();
            this.layerCollection = new Collection<Layer>();
            this.optimizer = optimizer;
            this.lossFunction = lossFunction;

            foreach (Layer layer in collection)
            {
                if (this.layerCollection.Count > 0)
                {
                    var previousLayer = this.layerCollection[this.layerCollection.Count - 1];

                    previousLayer.Next = layer;
                    layer.Previous = previousLayer;
                }

                this.layerCollection.Add(layer);
            }
        }

        public void Fit(IEnumerable<Tuple<double[], double[]>> collection, int epochs, int batchSize = 32)
        {
            Fit(collection, epochs, batchSize, (x, y) => x.Sample<Tuple<double[], double[]>>(this.random, y));
        }

        public void Fit(IEnumerable<Tuple<double[], double[]>> collection, int epochs, int batchSize, Func<IEnumerable<Tuple<double[], double[]>>, int, IEnumerable<Tuple<double[], double[]>>> func)
        {
            // Backpropagation
            int dataSize = collection.Count();
            int t = 0;

            // Stochastic gradient descent (SGD)
            while (t < epochs)
            {
                // Mini-batch
                int remaining = dataSize;

                do
                {
                    var dataTuple = func(collection, Math.Min(remaining, batchSize)).Aggregate<Tuple<double[], double[]>, Tuple<List<double[]>, List<double[]>>>(Tuple.Create<List<double[]>, List<double[]>>(new List<double[]>(), new List<double[]>()), (tuple1, tuple2) =>
                    {
                        tuple1.Item1.Add(tuple2.Item1);
                        tuple1.Item2.Add(tuple2.Item2);

                        return tuple1;
                    });
                    int index = 0;
                    int identifier = 0;
                    var tuples = Backward(Forward(new Batch<double[]>(dataTuple.Item1), true), new Batch<double[]>(dataTuple.Item2));

                    foreach (var tuple in tuples)
                    {
                        tuple.SetGradients((x, y, z) => x ? y + this.weightDecayRate * tuple.Weights[z] : y);
                    }

                    if (this.maxGradient.HasValue)
                    {
                        // Gradients clipping
                        var vectors = from tuple in tuples let batch = tuple.GetGradients() from vector in batch select vector;
                        double sum = 0.0;

                        foreach (var gradient in from vector in vectors from gradient in vector select gradient)
                        {
                            sum += gradient * gradient;
                        }

                        double rate = this.maxGradient.Value / (Math.Sqrt(sum) + Math.Pow(10, -6));

                        if (rate < 1)
                        {
                            foreach (var vector in vectors)
                            {
                                for (int i = 0; i < vector.Length; i++)
                                {
                                    vector[i] *= rate;
                                }
                            }
                        }
                    }

                    foreach (var tuple in tuples)
                    {
                        tuple.Update(tuple.GetGradients(), (weight, gradient) => optimizer.Optimize(identifier++, weight, gradient));
                        index++;
                    }

                    remaining -= batchSize;
                } while (remaining > 0);

                this.loss = GetLoss(collection);

                if (this.Stepped != null)
                {
                    this.Stepped(this, new EventArgs());
                }

                t++;
            }
        }

        public double[] Predicate(double[] vector)
        {
            var inputs = new Batch<double[]>(new double[][] { vector });
            var layer = this.layerCollection[0];
            Layer outputLayer;

            do
            {
                inputs = layer.Forward(inputs, false);

                outputLayer = layer;
                layer = layer.Next;
            } while (layer != null);

            return inputs[0];
        }

        private double GetLoss(IEnumerable<Tuple<double[], double[]>> collection)
        {
            double sum = 0.0;
            int size = collection.Count();
            int outputs = this.layerCollection[this.layerCollection.Count - 1].Outputs;

            foreach (var tuple in collection)
            {
                var outputActivations = Forward(new Batch<double[]>(new double[][] { tuple.Item1 }), false);

                for (int i = 0; i < outputs; i++)
                {
                    sum += this.lossFunction.Function(outputActivations[0][i], tuple.Item2[i]);
                }
            }

            return sum / size;
        }

        private Batch<double[]> Forward(Batch<double[]> x, bool isTraining)
        {
            var layer = this.layerCollection[0];
            var weightDecay = 0.0;

            do
            {
                var updatable = layer as IUpdatable;

                x = layer.Forward(x, isTraining);

                if (updatable != null)
                {
                    var sum = 0.0;

                    foreach (double weight in updatable.Weights)
                    {
                        sum = + weight * weight;
                    }

                    weightDecay += 0.5 * this.weightDecayRate * sum;
                }

                layer = layer.Next;
            } while (layer != null);

            foreach (var vector in x)
            {
                for (int i = 0; i < vector.Length; i++)
                {
                    vector[i] += weightDecay;
                }
            }

            return x;
        }

        private IEnumerable<IUpdatable> Backward(Batch<double[]> y, Batch<double[]> t)
        {
            var layer = this.layerCollection[this.layerCollection.Count - 1];
            var deltas = new Batch<double[]>(new double[t.Size][]);
            var updatableList = new LinkedList<IUpdatable>();

            for (int i = 0; i < t.Size; i++)
            {
                deltas[i] = new double[layer.Outputs];

                for (int j = 0; j < layer.Outputs; j++)
                {
                    deltas[i][j] = this.lossFunction.Derivative(y[i][j], t[i][j]);
                }
            }

            do
            {
                var updatable = layer as IUpdatable;

                deltas = layer.Backward(deltas);

                if (updatable != null)
                {
                    updatableList.AddFirst(updatable);
                }

                layer = layer.Previous;
            } while (layer != null);

            return updatableList;
        }
    }
}
