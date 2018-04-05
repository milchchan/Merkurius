using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Threading.Tasks;
using System.Diagnostics;
using Megalopolis;
using Megalopolis.ActivationFunctions;
using Megalopolis.Layers;
using Megalopolis.LossFunctions;
using Megalopolis.Optimizers;

namespace MegalopolisTest
{
    class Program
    {
        static void Main(string[] args)
        {
            int seed;

            using (var rng = new RNGCryptoServiceProvider())
            {
                var buffer = new byte[sizeof(int)];

                rng.GetBytes(buffer);
                seed = BitConverter.ToInt32(buffer, 0);
            }

            RandomProvider.SetSeed(seed);
            //RandomProvider.SetSeed(1900740164);

            var random = RandomProvider.GetRandom();
            var patternDictionary = new Dictionary<double[], IEnumerable<double[]>>();
            var logDictionary = new Dictionary<string, IEnumerable<double>>();

            patternDictionary.Add(new double[] { 0 }, new double[][] { new double[] { 0, 0 } });
            patternDictionary.Add(new double[] { 1 }, new double[][] { new double[] { 0, 1 } });
            patternDictionary.Add(new double[] { 1 }, new double[][] { new double[] { 1, 0 } });
            patternDictionary.Add(new double[] { 0 }, new double[][] { new double[] { 1, 1 } });

            Console.WriteLine("XOR Test ({0})", seed);

            var sw = Stopwatch.StartNew();
            var keyValuePairList = patternDictionary.Aggregate<KeyValuePair<double[], IEnumerable<double[]>>, List<KeyValuePair<double[], double[]>>>(new List<KeyValuePair<double[], double[]>>(), (list, kvp) =>
            {
                foreach (var vector in kvp.Value)
                {
                    list.Add(new KeyValuePair<double[], double[]>(vector, kvp.Key));
                }

                return list;
            });
            var accuracyList = new List<double>();
            var lossList = new List<double>();
            Func<int, int, double> weightFunc = (x, y) =>
            {
                var a = 4 * Math.Sqrt(6 / (x + y));

                return random.Uniform(-a, a);
            };
            var network = new Network(new FullyConnectedLayer(2, new Sigmoid(), x => weightFunc(2, 2), new FullyConnectedLayer(2, 1, new Sigmoid(), x => weightFunc(2, 1))), new AdaDelta(), new MeanSquaredError());
            /*Func<int, int, double> weightFunc = (x, y) =>
            {
                var a = Math.Sqrt(6 / (x + y));

                return random.Uniform(-a, a);
            };
            var network = new Network(
                new ConvolutionalPoolingLayer(12, 12, 1, 10, 3, 3, 2, 2, new ReLU(), x => weightFunc(20, 20),
                new ConvolutionalPoolingLayer(12, 1, 1, 20, 2, 2, 2, 2, new ReLU(), x => weightFunc(20, 20),
                new FullyConnectedLayer(20, new ReLU(), x => weightFunc(20, 20),
                new SoftmaxLayer(20, 3, x => weightFunc(20, 3))))),
                new SGD(), new CategoricalCrossEntropy());*/

            network.Stepped += (sender, e) =>
            {
                double tptn = 0;

                keyValuePairList.ForEach(kvp =>
                {
                    var vector = network.Predicate(kvp.Key);
                    var i = ArgMax(vector);
                    var j = ArgMax(kvp.Value);

                    if (i == j && Math.Round(vector[i]) == kvp.Value[j])
                    {
                        tptn += 1.0;
                    }
                });

                accuracyList.Add(tptn / keyValuePairList.Count);
                lossList.Add(network.Loss);
            };

            sw.Reset();

            Console.Write("Training...");

            sw.Start();

            network.Train(patternDictionary, 1000000);

            sw.Stop();

            Console.WriteLine("Done ({0}).", sw.Elapsed.ToString());
            Console.WriteLine();

            foreach (var vector in patternDictionary.Values.Aggregate<IEnumerable<double[]>, List<double[]>>(new List<double[]>(), (list, vectors) =>
            {
                foreach (var vector in vectors)
                {
                    list.Add(vector);
                }

                return list;
            }))
            {
                Console.WriteLine("{0}->{1}", String.Join(",", vector.Aggregate<double, List<string>>(new List<string>(), (x, y) =>
                {
                    x.Add(y.ToString());

                    return x;
                })), String.Join(",", network.Predicate(vector).Aggregate<double, List<string>>(new List<string>(), (x, y) =>
                {
                    x.Add(y.ToString());

                    return x;
                })));
            }

            Console.WriteLine();
            Console.WriteLine("Accuracy: {0}", accuracyList.Last());
            Console.WriteLine("Loss: {0}", lossList.Last());
            /*
            logDictionary.Add("Accuracy", accuracyList);
            logDictionary.Add("Loss", lossList);

            var path = "Log.csv";

            Console.WriteLine();
            Console.Write("Writing to {0}...", path);

            ToCsv(path, logDictionary);

            Console.WriteLine("Done.");*/
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

        static private void ToCsv(string path, Dictionary<string, IEnumerable<double>> dictionary)
        {
            using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read))
            using (var s = new StreamWriter(fs, System.Text.Encoding.UTF8))
            {
                s.Write(String.Join(",", dictionary.Keys));
                s.Write("\r\n");

                var values = dictionary.Values.ToArray();
                var table = new List<List<string>>();
                int maxLines = 0;

                foreach (var data in dictionary.Values)
                {
                    var array = data.ToArray();
                    var list = new List<string>();

                    if (array.Length > maxLines)
                    {
                        maxLines = array.Length;
                    }

                    foreach (var item in array)
                    {
                        list.Add(item.ToString());
                    }

                    table.Add(list);
                }

                for (int i = 0; i < table.Count; i++)
                {
                    for (int j = table[i].Count; table[i].Count < maxLines; j++)
                    {
                        table[i].Add(String.Empty);
                    }
                }

                for (int i = 0; i < maxLines; i++)
                {
                    var dataList = new List<string>();

                    for (int j = 0; j < values.Length; j++)
                    {
                        dataList.Add(table[j][i].ToString());
                    }

                    s.Write(String.Join(",", dataList));
                    s.Write("\r\n");
                }

                fs.Flush();
            }
        }
    }
}
