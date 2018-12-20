using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Megalopolis
{
    namespace Layers
    {
        public class Dropout : IFilter
        {
            private double rate = 0.5;
            private double[][] masks = null;

            public double Rate
            {
                get
                {
                    return this.rate;
                }
            }

            public Dropout() { }

            public Dropout(double rate)
            {
                this.rate = rate;
            }

            public Batch<double[]> Forward(Batch<double[]> batch, bool isTraining)
            {
                if (isTraining)
                {
                    var parallelOptions = new ParallelOptions();
                    var tuple = Tuple.Create<double[][], double[][]>(new double[batch.Size][], new double[batch.Size][]);

                    parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                    Parallel.ForEach<double[], List<Tuple<long, double[], double[]>>>(batch, parallelOptions, () => new List<Tuple<long, double[], double[]>>(), (vector1, state, index, local) =>
                    {
                        Random random = random = RandomProvider.GetRandom(); ;
                        double[] masks = new double[vector1.Length];
                        double[] vector2 = new double[vector1.Length];

                        for (int i = 0; i < vector1.Length; i++)
                        {
                            double probability = random.Binomial(1, this.rate);

                            masks[i] = probability;
                            vector2[i] = vector1[i] * probability;
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

                return batch;
            }

            public Batch<double[]> Backward(Batch<double[]> batch)
            {
                var parallelOptions = new ParallelOptions();
                var data = new double[batch.Size][];
                List<double[]> vectorList = new List<double[]>();

                parallelOptions.MaxDegreeOfParallelism = 2 * Environment.ProcessorCount;

                Parallel.ForEach<double[], List<Tuple<long, double[]>>>(batch, parallelOptions, () => new List<Tuple<long, double[]>>(), (vector1, state, index, local) =>
                {
                    double[] vector2 = new double[vector1.Length];

                    for (int i = 0; i < vector1.Length; i++)
                    {
                        vector2[i] = vector1[i] * this.masks[index][i];
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
