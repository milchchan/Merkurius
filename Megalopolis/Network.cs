using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
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
        private Layer outputLayer = null;
        private Collection<Layer> layerCollection = null;
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

        public Network(Layer inputLayer, Layer outputLayer, IOptimizer optimizer, ILossFunction lossFunction)
        {
            var layer = inputLayer;

            this.random = RandomProvider.GetRandom();
            this.inputLayer = inputLayer;
            this.outputLayer = outputLayer;
            this.layerCollection = new Collection<Layer>();
            this.optimizer = optimizer;
            this.lossFunction = lossFunction;

            do
            {
                this.layerCollection.Add(layer);
                layer = layer.Next;
            } while (layer != null);
        }

        public void Train(IEnumerable<Tuple<double[], double[]>> collection, int epochs, int batchSize = 32)
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

                    foreach (var gradients in BackwardPropagate(ForwardPropagate(new Batch<double[]>(dataTuple.Item1), true), new Batch<double[]>(dataTuple.Item2)))
                    {
                        this.layerCollection[index].Update(gradients, (weight, gradient) => optimizer.Optimize(identifier++, weight, gradient));
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
                inputs = layer.PropagateForward(inputs, false);

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
                var activations = ForwardPropagate(new Batch<double[]>(new double[][] { tuple.Item1 }), false);
                var outputActivations = activations.Last().Item2;

                for (int i = 0; i < this.outputLayer.Outputs; i++)
                {
                    sum += this.lossFunction.Function(outputActivations[0][i], tuple.Item2[i]);
                }
            }

            return sum / size;
        }

        private IEnumerable<Tuple<Batch<double[]>, Batch<double[]>>> ForwardPropagate(Batch<double[]> inputs, bool isTraining)
        {
            var layer = this.inputLayer;
            var tupleList = new List<Tuple<Batch<double[]>, Batch<double[]>>>();

            do
            {
                var outputs = layer.PropagateForward(inputs, isTraining);
                
                tupleList.Add(Tuple.Create<Batch<double[]>, Batch<double[]>>(inputs, outputs));
                inputs = outputs;

                layer = layer.Next;
            } while (layer != null);

            return tupleList;
        }

        private IEnumerable<Batch<double[]>> BackwardPropagate(IEnumerable<Tuple<Batch<double[]>, Batch<double[]>>> activations, Batch<double[]> outputs)
        {
            var layer = this.outputLayer.Previous;
            var activationsLinkedList = new LinkedList<Tuple<Batch<double[]>, Batch<double[]>>>(activations);
            var deltas = new Batch<double[]>(new double[outputs.Size][]);
            var gradientsList = new LinkedList<Batch<double[]>>();

            for (int i = 0; i < outputs.Size; i++)
            {
                deltas[i] = new double[this.outputLayer.Outputs];

                for (int j = 0; j < this.outputLayer.Outputs; j++)
                {
                    deltas[i][j] = this.lossFunction.Derivative(activationsLinkedList.Last.Value.Item2[i][j], outputs[i][j]);
                }
            }

            var tuple = this.outputLayer.PropagateBackward(activationsLinkedList.Last.Value.Item1, activationsLinkedList.Last.Value.Item2, deltas);

            gradientsList.AddFirst(tuple.Item2);
            activationsLinkedList.RemoveLast();

            while (layer != null)
            {
                tuple = layer.PropagateBackward(activationsLinkedList.Last.Value.Item1, activationsLinkedList.Last.Value.Item2, tuple.Item1);

                gradientsList.AddFirst(tuple.Item2);
                activationsLinkedList.RemoveLast();

                layer = layer.Previous;
            }

            return gradientsList;
        }
    }
}
