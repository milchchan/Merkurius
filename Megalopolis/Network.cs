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

        public Network(Layer layer, IOptimizer optimizer, ILossFunction lossFunction)
        {
            this.random = RandomProvider.GetRandom();
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

        public void Train(IDictionary<double[], IEnumerable<double[]>> dictionary, int epochs, int batchSize = 32)
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
            int maxThreads = 2 * Environment.ProcessorCount;
            int t = 0;

            // Stochastic gradient descent (SGD)
            while (t < epochs)
            {
                // Mini-batch
                int remaining = keyValuePairList.Count;

                do
                {
                    var batchDataQueue = new Queue<Tuple<Layer, double[], double[]>>(keyValuePairList.Sample<KeyValuePair<double[], double[]>>(this.random, Math.Min(remaining, batchSize)).Aggregate<KeyValuePair<double[], double[]>, List<Tuple<Layer, double[], double[]>>>(new List<Tuple<Layer, double[], double[]>>(), (list, keyValuePair) =>
                    {
                        list.Add(Tuple.Create<Layer, double[], double[]>(Copy(this.inputLayer), keyValuePair.Key, keyValuePair.Value));

                        return list;
                    }));
                    var batchTaskList = new List<Task<IEnumerable<Tuple<double[], double[]>>>>();
                    var batchedDataList = new LinkedList<IEnumerable<Tuple<double[], double[]>>>();
                    var mergedDataTuples = new Tuple<double[], double[]>[this.layerCollection.Count];
                    int index = 0;

                    do
                    {
                        do
                        {
                            var task = new Task<IEnumerable<Tuple<double[], double[]>>>(delegate (object state)
                            {
                                var tuple = (Tuple<Layer, double[], double[]>)state;

                                return BackwardPropagate(ForwardPropagate(true, tuple.Item1, tuple.Item2), tuple.Item3);
                            }, batchDataQueue.Dequeue());

                            batchTaskList.Add(task);
                            task.Start();
                        } while (batchDataQueue.Count > 0 && batchTaskList.Count < maxThreads);

                        var tasks = batchTaskList.ToArray();
                        var i = Task<IEnumerable<Tuple<double[], double[]>>>.WaitAny(tasks);

                        for (int j = 0; j < tasks.Length; j++)
                        {
                            if (i == j)
                            {
                                if (tasks[j].Exception == null)
                                {
                                    batchedDataList.AddLast(tasks[j].Result);
                                }

                                batchTaskList.RemoveAt(i);

                                break;
                            }
                        }
                    } while (batchDataQueue.Count > 0);

                    if (batchTaskList.Count > 0)
                    {
                        var tasks = batchTaskList.ToArray();

                        Task<IEnumerable<Tuple<double[], double[]>>>.WaitAll(tasks);

                        foreach (var task in tasks)
                        {
                            if (task.Exception == null)
                            {
                                batchedDataList.AddLast(task.Result);
                            }
                        }
                    }

                    foreach (var (gradients, deltas) in batchedDataList.First.Value)
                    {
                        mergedDataTuples[index] = Tuple.Create<double[], double[]>(new double[gradients.Length], new double[deltas.Length]);

                        for (int j = 0; j < gradients.Length; j++)
                        {
                            mergedDataTuples[index].Item1[j] = gradients[j];
                        }

                        for (int j = 0; j < deltas.Length; j++)
                        {
                            mergedDataTuples[index].Item2[j] = deltas[j];
                        }

                        index++;
                    }

                    for (var tuplesNode = batchedDataList.First.Next; tuplesNode != null; tuplesNode = tuplesNode.Next)
                    {
                        index = 0;

                        foreach (var (gradients, deltas) in tuplesNode.Value)
                        {
                            for (int j = 0; j < gradients.Length; j++)
                            {
                                mergedDataTuples[index].Item1[j] += gradients[j];
                            }

                            for (int j = 0; j < deltas.Length; j++)
                            {
                                mergedDataTuples[index].Item2[j] += deltas[j];
                            }

                            index++;
                        }
                    }

                    for (int i = 0, j = 0; i < this.layerCollection.Count; i++)
                    {
                        for (int k = 0; k < mergedDataTuples[i].Item1.Length; k++)
                        {
                            mergedDataTuples[i].Item1[k] = mergedDataTuples[i].Item1[k] / batchedDataList.Count;
                        }

                        for (int k = 0; k < mergedDataTuples[i].Item2.Length; k++)
                        {
                            mergedDataTuples[i].Item2[k] = mergedDataTuples[i].Item2[k] / batchedDataList.Count;
                        }

                        this.layerCollection[i].Update(mergedDataTuples[i].Item1, mergedDataTuples[i].Item2, (weight, gradient) => optimizer.Optimize(j++, weight, gradient));
                    }

                    remaining -= batchSize;
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

        private IEnumerable<Tuple<double[], double[]>> BackwardPropagate(Layer outputLayer, double[] vector)
        {
            var layer = outputLayer.Previous;
            var deltas = new double[outputLayer.OutputActivations.Length];
            var deltasList = new LinkedList<double[]>();
            double[] gradients;
            var gradientsList = new LinkedList<double[]>();
            int length = 1;
            var tupleList = new List<Tuple<double[], double[]>>();

            for (int i = 0; i < outputLayer.OutputActivations.Length; i++)
            {
                deltas[i] = this.lossFunction.Derivative(outputLayer.OutputActivations[i], vector[i]);
            }

            foreach (var g in outputLayer.PropagateBackward(ref deltas, out gradients))
            {
                deltasList.AddFirst(g);
            }

            gradientsList.AddFirst(gradients);
            deltas = deltasList.First.Value;

            while (layer != null)
            {
                var tempDeltasList = new LinkedList<double[]>(layer.PropagateBackward(ref deltas, out gradients));

                gradientsList.AddFirst(gradients);
                deltasList.First.Value = deltas;
                deltas = tempDeltasList.Last.Value;

                foreach (var g in tempDeltasList)
                {
                    deltasList.AddFirst(g);
                }

                layer = layer.Previous;
                length++;
            }

            deltasList.RemoveFirst();

            var gradientsNode = gradientsList.First;
            var deltasNode = deltasList.First;

            for (int i = 0; i < length; i++)
            {
                tupleList.Add(Tuple.Create<double[], double[]>(gradientsNode.Value, deltasNode.Value));

                gradientsNode = gradientsNode.Next;
                deltasNode = deltasNode.Next;
            }

            return tupleList;
        }

        private Layer Copy(Layer inputLayer)
        {
            var layer = inputLayer;

            while (layer.Next != null)
            {
                layer = layer.Next;
            }

            var copiedLayer = layer.Copy();

            while (layer.Previous != null)
            {
                copiedLayer = layer.Previous.Copy(copiedLayer);
                layer = layer.Previous;
            }

            return copiedLayer;
        }
    }
}
