using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Merkurius.Optimizers;
using Merkurius.Layers;
using Merkurius.LossFunctions;

namespace Merkurius
{
    public class Model
    {
        public event EventHandler<EventArgs> Stepped = null;
        private Random random = null;
        private Collection<Layer> layerCollection = null;
        private double weightDecayRate = 0.0;
        private double? maxGradient = null;

        public IEnumerable<Layer> Layers
        {
            get
            {
                return this.layerCollection;
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

        public Model(Layer inputLayer)
        {
            var layer = inputLayer;

            this.random = RandomProvider.GetRandom();
            this.layerCollection = new Collection<Layer>();

            do
            {
                this.layerCollection.Add(layer);
                layer = layer.Next;
            } while (layer != null);
        }

        public Model(IEnumerable<Layer> collection)
        {
            this.random = RandomProvider.GetRandom();
            this.layerCollection = new Collection<Layer>();

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

        public void Fit(IEnumerable<ValueTuple<double[], double[]>> collection, int epochs, int batchSize, IOptimizer optimizer, ILossFunction lossFunction)
        {
            Fit(collection, epochs, batchSize, (x, y) => x.Sample<ValueTuple<double[], double[]>>(this.random, y), optimizer, lossFunction);
        }

        public void Fit(IEnumerable<ValueTuple<double[], double[]>> collection, int epochs, int batchSize, Func<IEnumerable<ValueTuple<double[], double[]>>, int, IEnumerable<ValueTuple<double[], double[]>>> func, IOptimizer optimizer, ILossFunction lossFunction)
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
                    var dataTuple = func(collection, Math.Min(remaining, batchSize)).Aggregate<ValueTuple<double[], double[]>, Tuple<List<double[]>, List<double[]>>>(Tuple.Create<List<double[]>, List<double[]>>(new List<double[]>(), new List<double[]>()), (tuple1, tuple2) =>
                    {
                        tuple1.Item1.Add(tuple2.Item1);
                        tuple1.Item2.Add(tuple2.Item2);

                        return tuple1;
                    });
                    int index = 0;
                    int identifier = 0;
                    var targets = new Batch<double[]>(dataTuple.Item2);
                    var layers = Backward(Forward(new Batch<double[]>(dataTuple.Item1), targets, true, lossFunction).Item1, targets, lossFunction);

                    // Weight decay
                    foreach (var layer in layers)
                    {
                        layer.SetGradients((x, y, z) => x ? y + this.weightDecayRate * layer.Weights[z] : y);
                    }

                    if (this.maxGradient.HasValue)
                    {
                        // Gradient clipping
                        var vectors = from tuple in layers let batch = tuple.GetGradients() from vector in batch select vector;
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

                    foreach (var layer in layers)
                    {
                        layer.Update(layer.GetGradients(), (weight, gradient) => optimizer.Optimize(identifier++, weight, gradient));
                        index++;
                    }

                    remaining -= batchSize;
                } while (remaining > 0);

                if (this.Stepped != null)
                {
                    this.Stepped(this, new EventArgs());
                }

                t++;
            }
        }

        public double[] Predict(double[] vector)
        {
            var inputs = new Batch<double[]>(new double[][] { vector });
            var layer = this.layerCollection[0];

            do
            {
                inputs = layer.Forward(inputs, false);
                layer = layer.Next;
            } while (layer != null);

            return inputs[0];
        }

        public double GetLoss(IEnumerable<ValueTuple<double[], double[]>> collection, ILossFunction lossFunction)
        {
            double sum = 0.0;
            int size = collection.Count();
            int outputs = this.layerCollection[this.layerCollection.Count - 1].Outputs;

            foreach (var loss in from tuple in collection from loss in Forward(new Batch<double[]>(new double[][] { tuple.Item1 }), new Batch<double[]>(new double[][] { tuple.Item2 }), false, lossFunction).Item2[0] select loss)
            {
                sum += loss;
            }

            return sum / size;
        }

        private Tuple<Batch<double[]>, Batch<double[]>> Forward(Batch<double[]> x, Batch<double[]> t, bool isTraining, ILossFunction lossFunction)
        {
            var layer = this.layerCollection[0];
            var weightDecay = 0.0;
            var vectorList1 = new List<double[]>();
            var vectorList2 = new List<double[]>();

            do
            {
                var updatable = layer as IUpdatable;

                x = layer.Forward(x, isTraining);

                if (updatable != null)
                {
                    var sum = 0.0;

                    foreach (double weight in updatable.Weights)
                    {
                        sum += weight * weight;
                    }

                    weightDecay += 0.5 * this.weightDecayRate * sum;
                }

                layer = layer.Next;
            } while (layer != null);

            for (int i = 0; i < x.Size; i++)
            {
                var y = lossFunction.Forward(x[i], t[i]);

                for (int j = 0; j < y.Item2.Length; j++)
                {
                    y.Item2[j] += weightDecay;
                }

                vectorList1.Add(y.Item1);
                vectorList2.Add(y.Item2);
            }

            return Tuple.Create<Batch<double[]>, Batch<double[]>>(new Batch<double[]>(vectorList1), new Batch<double[]>(vectorList2));
        }

        private IEnumerable<IUpdatable> Backward(Batch<double[]> y, Batch<double[]> t, ILossFunction lossFunction)
        {
            var layer = this.layerCollection[this.layerCollection.Count - 1];
            var deltas = new Batch<double[]>(new double[t.Size][]);
            var updatableList = new LinkedList<IUpdatable>();

            for (int i = 0; i < t.Size; i++)
            {
                deltas[i] = lossFunction.Backward(y[i], t[i]);
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
