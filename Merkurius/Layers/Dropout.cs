using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace Merkurius
{
    namespace Layers
    {
        [DataContract]
        public class Dropout : Layer
        {
            [DataMember]
            private double rate = 0.5;
            [DataMember]
            private double[][]? masks = null;

            public double Rate
            {
                get
                {
                    return this.rate;
                }
            }

            public Dropout(int nodes) : base(nodes, nodes) { }

            public Dropout(int nodes, double rate) : base(nodes, nodes)
            {
                this.rate = rate;
            }

            public Dropout(double rate, Layer layer) : base(layer.Inputs, layer)
            {
                this.rate = rate;
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                var parallelOptions = new ParallelOptions();

                if (isTraining)
                {
                    var tuple = Tuple.Create<double[][], double[][]>(new double[inputs.Size][], new double[inputs.Size][]);

                    parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                    Parallel.ForEach<double[], List<Tuple<long, double[], double[]>>>(inputs, parallelOptions, () => new List<Tuple<long, double[], double[]>>(), (vector1, state, index, local) =>
                    {
                        Random random = random = RandomProvider.GetRandom();
                        double[] masks = new double[vector1.Length];
                        double[] vector2 = new double[vector1.Length];

                        for (int i = 0; i < vector1.Length; i++)
                        {
                            if (random.NextDouble() > this.rate)
                            {
                                masks[i] = 1.0;
                                vector2[i] = vector1[i];
                            }
                            else
                            {
                                masks[i] = 0.0;
                                vector2[i] = 0.0;
                            }
                        }

                        local.Add(Tuple.Create<long, double[], double[]>(index, masks, vector2));

                        return local;
                    }, (local) =>
                    {
                        lock (tuple)
                        {
                            local.ForEach(x =>
                            {
                                tuple.Item1[x.Item1] = x.Item2;
                                tuple.Item2[x.Item1] = x.Item3;
                            });
                        }
                    });

                    this.masks = tuple.Item1;

                    return new Batch<double[]>(tuple.Item2);
                }

                var data = new double[inputs.Size][];

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(inputs, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector1, state, index, local) =>
                {
                    double[] vector2 = new double[vector1.Length];

                    for (int i = 0; i < vector1.Length; i++)
                    {
                        vector2[i] = vector1[i] * (1.0 - this.rate);
                    }

                    local.Add(Tuple.Create<long, double[]>(index, vector2));

                    return local;
                }, (local) =>
                {
                    lock (data)
                    {
                        local.ForEach(x =>
                        {
                            data[x.Item1] = x.Item2;
                        });
                    }
                });

                return new Batch<double[]>(data); ;
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                var parallelOptions = new ParallelOptions();
                var data = new double[deltas.Size][];

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(deltas, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector1, state, index, local) =>
                {
                    double[] vector2 = new double[vector1.Length];

                    for (int i = 0; i < vector1.Length; i++)
                    {
                        vector2[i] = vector1[i] * this.masks![index][i];
                    }

                    local.Add(Tuple.Create<long, double[]>(index, vector2));

                    return local;
                }, (local) =>
                {
                    lock (data)
                    {
                        local.ForEach(x =>
                        {
                            data[x.Item1] = x.Item2;
                        });
                    }
                });

                return new Batch<double[]>(data);
            }
        }
    }
}
