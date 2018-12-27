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
        private Layer inputLayer = null;
        private Layer outputLayer = null;
        private Collection<Layer> layerCollection = null;
        private double loss = 0.0;
        private IOptimizer optimizer = null;
        private ILossFunction lossFunction = null;
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

        public Model(Layer outputLayer, IOptimizer optimizer, ILossFunction lossFunction)
        {
            var layer = outputLayer;

            this.random = RandomProvider.GetRandom();
            this.outputLayer = outputLayer;
            this.layerCollection = new Collection<Layer>();
            this.optimizer = optimizer;
            this.lossFunction = lossFunction;

            do
            {
                this.inputLayer = layer;
                this.layerCollection.Insert(0, layer);
                layer = layer.Previous;
            } while (layer != null);
        }

        public void Fit(IEnumerable<Tuple<double[], double[]>> collection, int epochs, int batchSize = 32)
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
                    var dataTuple = collection.Sample<Tuple<double[], double[]>>(this.random, Math.Min(remaining, batchSize)).Aggregate<Tuple<double[], double[]>, Tuple<List<double[]>, List<double[]>>>(Tuple.Create<List<double[]>, List<double[]>>(new List<double[]>(), new List<double[]>()), (tuple1, tuple2) =>
                    {
                        tuple1.Item1.Add(tuple2.Item1);
                        tuple1.Item2.Add(tuple2.Item2);

                        return tuple1;
                    });
                    int index = 0;
                    int identifier = 0;
                    var tuples = Backward(Forward(new Batch<double[]>(dataTuple.Item1), true), new Batch<double[]>(dataTuple.Item2));

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
            var layer = this.inputLayer;
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

            foreach (var tuple in collection)
            {
                var outputActivations = Forward(new Batch<double[]>(new double[][] { tuple.Item1 }), false);

                for (int i = 0; i < this.outputLayer.Outputs; i++)
                {
                    sum += this.lossFunction.Function(outputActivations[0][i], tuple.Item2[i]);
                }
            }

            return sum / size;
        }

        private Batch<double[]> Forward(Batch<double[]> x, bool isTraining)
        {
            var layer = this.inputLayer;

            do
            {
                x = layer.Forward(x, isTraining);

                layer = layer.Next;
            } while (layer != null);

            return x;
        }

        private IEnumerable<IUpdatable> Backward(Batch<double[]> y, Batch<double[]> t)
        {
            var layer = this.outputLayer;
            var deltas = new Batch<double[]>(new double[t.Size][]);
            var updatableList = new LinkedList<IUpdatable>();

            for (int i = 0; i < t.Size; i++)
            {
                deltas[i] = new double[this.outputLayer.Outputs];

                for (int j = 0; j < this.outputLayer.Outputs; j++)
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
