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
            Console.WriteLine();

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
            var logDictionary = new Dictionary<string, IEnumerable<double>>();

            patternList.Add(Tuple.Create<double[], double[]>(new double[] { 0, 0 }, new double[] { 0 }));
            patternList.Add(Tuple.Create<double[], double[]>(new double[] { 0, 1 }, new double[] { 1 }));
            patternList.Add(Tuple.Create<double[], double[]>(new double[] { 1, 0 }, new double[] { 1 }));
            patternList.Add(Tuple.Create<double[], double[]>(new double[] { 1, 1 }, new double[] { 0 }));

            var stopwatch = Stopwatch.StartNew();
            var accuracyList = new List<double>();
            var lossList = new List<double>();
            var network = new Network(new FullyConnectedLayer(2, new Sigmoid(), (index, fanIn, fanOut) => RandomProvider.GetRandom().NextDouble(), new FullyConnectedLayer(2, 1, new Sigmoid(), (index, fanIn, fanOut) => RandomProvider.GetRandom().NextDouble())), new Momentum(), new MeanSquaredError());
            int epochs = 1;

            network.Stepped += (sender, e) =>
            {
                double tptn = 0;

                patternList.ForEach(tuple =>
                {
                    var vector = network.Predicate(tuple.Item1);
                    var i = ArgMax(vector);
                    var j = ArgMax(tuple.Item2);

                    if (i == j && Math.Round(vector[i]) == tuple.Item2[j])
                    {
                        tptn += 1.0;
                    }
                });

                var accuracy = tptn / patternList.Count;

                accuracyList.Add(accuracy);
                lossList.Add(network.Loss);

                if (epochs % 1000 == 0)
                {
                    Console.WriteLine("Epochs: {0} (Accuracy: {1} / Loss: {2})", epochs, accuracy, network.Loss);
                }
                
                epochs++;
            };

            stopwatch.Reset();

            Console.WriteLine("Training...");

            stopwatch.Start();

            network.Train(patternList, 10000);

            stopwatch.Stop();

            Console.WriteLine("Done ({0}).", stopwatch.Elapsed.ToString());
            Console.WriteLine();

            foreach (var tuple in patternList)
            {
                Console.WriteLine("{0}->{1}", String.Join(",", tuple.Item1.Aggregate<double, List<string>>(new List<string>(), (x, y) =>
                {
                    x.Add(y.ToString());

                    return x;
                })), String.Join(",", network.Predicate(tuple.Item1).Aggregate<double, List<string>>(new List<string>(), (x, y) =>
                {
                    x.Add(y.ToString());

                    return x;
                })));
            }

            Console.WriteLine();
            Console.WriteLine("Accuracy: {0}", accuracyList.Last());
            Console.WriteLine("Loss: {0}", lossList.Last());
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
