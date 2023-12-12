using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace Merkurius.Layers
{
    // Deep Residual Learning for Image Recognition
    // Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    // https://arxiv.org/abs/1512.03385
    [DataContract]
    public class Skip : Layer, IUpdatable
    {
        [DataMember]
        private Collection<Layer>? layerCollection = null;

        public double[] Weights
        {
            get
            {
                var weightList = new List<double>();

                foreach (var layer in this.layerCollection!)
                {
                    var updatable = layer as IUpdatable;

                    if (updatable != null)
                    {
                        weightList.AddRange(updatable.Weights);
                    }
                }

                return weightList.ToArray();
            }
            set
            {
                var index = 0;

                foreach (var layer in this.layerCollection!)
                {
                    var updatable = layer as IUpdatable;

                    if (updatable != null)
                    {
                        for (int i = 0; i < updatable.Weights.Length; i++)
                        {
                            updatable.Weights[i] = value[index];
                            index++;
                        }
                    }
                }
            }
        }

        public IEnumerable<Layer> Layers
        {
            get
            {
                return this.layerCollection!;
            }
        }
        
        public Skip(Layer inputLayer, Layer outputLayer) : base(inputLayer, outputLayer)
        {
            var layer = inputLayer;

            this.layerCollection = new Collection<Layer>();

            do
            {
                this.layerCollection.Add(layer);
                layer = layer.Next;
            } while (layer != null);
        }

        public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
        {
            var layer = this.layerCollection![0];
            var x = inputs;
            var parallelOptions = new ParallelOptions();
            var data = new double[inputs.Size][];

            parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

            do
            {
                inputs = layer.Forward(inputs, isTraining);
                layer = layer.Next;
            } while (layer != null);

            Parallel.ForEach<double[], List<Tuple<long, double[]>>>(inputs, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector, state, index, local) =>
            {
                var y = new double[vector.Length];

                for (int i = 0; i < vector.Length; i++)
                {
                    y[i] = vector[i] + x[index][i];
                }

                local.Add(Tuple.Create<long, double[]>(index, y));

                return local;
            }, (local) =>
            {
                lock (data)
                {
                    local.ForEach(tuple =>
                    {
                        data[tuple.Item1] = tuple.Item2;
                    });
                }
            });

            return new Batch<double[]>(data);
        }

        public override Batch<double[]> Backward(Batch<double[]> deltas)
        {
            var layer = this.layerCollection![this.layerCollection.Count - 1];
            var dx = deltas;
            var parallelOptions = new ParallelOptions();
            var data = new double[deltas.Size][];

            do
            {
                deltas = layer.Backward(deltas);
                layer = layer.Previous;
            } while (layer != null);

            Parallel.ForEach<double[], List<Tuple<long, double[]>>>(deltas, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector1, state, index, local) =>
            {
                var vector2 = new double[vector1.Length];

                for (int i = 0; i < vector1.Length; i++)
                {
                    vector2[i] = vector1[i] + dx[index][i];
                }

                local.Add(Tuple.Create<long, double[]>(index, vector2));

                return local;
            }, (local) =>
            {
                lock (data)
                {
                    local.ForEach(tuple =>
                    {
                        data[tuple.Item1] = tuple.Item2;
                    });
                }
            });

            return new Batch<double[]>(data);
        }

        public Batch<double[]> GetGradients()
        {
            var layer = this.layerCollection![0];
            List<double>[]? gradients = null;

            do
            {
                var updatable = layer as IUpdatable;

                if (updatable != null)
                {
                    var batch = updatable.GetGradients();
                    var index = 0;

                    if (gradients == null)
                    {
                        gradients = new List<double>[batch.Size];

                        foreach (double[] vector in batch)
                        {
                            gradients[index] = new List<double>(vector);
                            index++;
                        }
                    }
                    else
                    {
                        foreach (double[] vector in batch)
                        {
                            gradients[index].AddRange(vector);
                            index++;
                        }
                    }
                }

                layer = layer.Next;
            } while (layer != null);

            return new Batch<double[]>(gradients.Aggregate<List<double>, List<double[]>>(new List<double[]>(), (list1, list2) =>
            {
                list1.Add(list2.ToArray());

                return list1;
            }));
        }

        public void SetGradients(Func<bool, double, int, double> func)
        {
            foreach (var layer in this.layerCollection!)
            {
                var updatable = layer as IUpdatable;

                if (updatable != null)
                {
                    updatable.SetGradients(func);
                }
            }
        }

        public void Update(Batch<double[]> gradients, Func<double, double, double> func)
        {
            int i = 0;

            foreach (var layer in this.layerCollection!)
            {
                var updatable = layer as IUpdatable;

                if (updatable != null)
                {
                    var batch = updatable.GetGradients();
                    var tempGradients = new double[batch.Size][];
                    var j = 0;
                    var k = 0;
                    
                    foreach (double[] vector1 in batch)
                    {
                        double[] vector2 = new double[vector1.Length];

                        j = 0;

                        do
                        {
                            vector2[j] = gradients[k][i + j];
                            j++;
                        } while (j < vector1.Length);

                        tempGradients[k] = vector2;
                        k++;
                    }

                    i += j;

                    updatable.Update(new Batch<double[]>(tempGradients), func);
                }
            }
        }
    }
}
