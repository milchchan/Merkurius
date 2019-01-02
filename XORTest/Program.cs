using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography;
using Megalopolis;
using Megalopolis.ActivationFunctions;
using Megalopolis.Layers;
using Megalopolis.LossFunctions;
using Megalopolis.Optimizers;

namespace XORTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("XOR Test");

            int seed;

            using (var rng = new RNGCryptoServiceProvider())
            {
                var buffer = new byte[sizeof(int)];

                rng.GetBytes(buffer);
                seed = BitConverter.ToInt32(buffer, 0);
            }

            RandomProvider.SetSeed(seed);
            
            var random = RandomProvider.GetRandom();
            var patternList = new List<Tuple<double[], double[]>>();
            var accuracyList = new List<double>();
            var lossList = new List<double>();

            patternList.Add(Tuple.Create<double[], double[]>(new double[] { 0, 0 }, new double[] { 0 }));
            patternList.Add(Tuple.Create<double[], double[]>(new double[] { 0, 1 }, new double[] { 1 }));
            patternList.Add(Tuple.Create<double[], double[]>(new double[] { 1, 0 }, new double[] { 1 }));
            patternList.Add(Tuple.Create<double[], double[]>(new double[] { 1, 1 }, new double[] { 0 }));

            var model = new Model(new Layer[] {
                new FullyConnected(2, 2, (index, fanIn, fanOut) => RandomProvider.GetRandom().NextDouble()),
                new Activation(2, new Sigmoid()),
                new FullyConnected(2, 1, (index, fanIn, fanOut) => RandomProvider.GetRandom().NextDouble()),
                new Activation(1, new Identity())
            }, new Momentum(0.5, 0.1), new SoftmaxCrossEntropy());
            int epochs = 10000;
            int iterations = 1;

            model.Stepped += (sender, e) =>
            {
                double tptn = 0.0;

                patternList.ForEach(tuple =>
                {
                    var vector = model.Predicate(tuple.Item1);
                    var i = ArgMax(vector);
                    var j = ArgMax(tuple.Item2);

                    if (i == j && Math.Round(vector[i]) == tuple.Item2[j])
                    {
                        tptn += 1.0;
                    }
                });

                var accuracy = tptn / patternList.Count;

                accuracyList.Add(accuracy);
                lossList.Add(model.Loss);

                if (iterations % 2500 == 0)
                {
                    Console.WriteLine("Epoch {0}/{1}", iterations, epochs);
                    Console.WriteLine("Accuracy: {0}, Loss: {1}", accuracy, model.Loss);
                }

                iterations++;
            };

            Console.WriteLine("Training...");

            var stopwatch = Stopwatch.StartNew();

            model.Fit(patternList, epochs);

            stopwatch.Stop();

            Console.WriteLine("Done ({0}).", stopwatch.Elapsed.ToString());

            foreach (var tuple in patternList)
            {
                Console.WriteLine("{0}->{1}", String.Join(",", tuple.Item1.Aggregate<double, List<string>>(new List<string>(), (x, y) =>
                {
                    x.Add(y.ToString());

                    return x;
                })), String.Join(",", model.Predicate(tuple.Item1).Aggregate<double, List<string>>(new List<string>(), (x, y) =>
                {
                    x.Add(y.ToString());

                    return x;
                })));
            }
        }

        static private int ArgMax(double[] vector)
        {
            int index = 0;
            var max = Double.MinValue;

            for (int i = 0; i < vector.Length; i++)
            {
                if (vector[i] > max)
                {
                    max = vector[i];
                    index = i;
                }
            }

            return index;
        }
    }
}
