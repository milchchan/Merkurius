using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using Megalopolis;
using Megalopolis.ActivationFunctions;
using Megalopolis.Layers;
using Megalopolis.LossFunctions;
using Megalopolis.Optimizers;
using Mnist;

namespace MNISTTest
{
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            Console.WriteLine("MNIST Test");

            int seed;

            using (var rng = new RNGCryptoServiceProvider())
            {
                var buffer = new byte[sizeof(int)];

                rng.GetBytes(buffer);
                seed = BitConverter.ToInt32(buffer, 0);
            }

            RandomProvider.SetSeed(seed);

            var assembly = Assembly.GetExecutingAssembly();
            var random = RandomProvider.GetRandom();
            var trainingList = new List<Tuple<double[], double[]>>();
            var testList = new List<Tuple<double[], double[]>>();
            var accuracyList = new List<double>();
            var lossList = new List<double>();
            var logDictionary = new Dictionary<string, IEnumerable<double>>();
            var logPath = "Log.csv";
            var channels = 1;
            var imageWidth = 28;
            var imageHeight = 28;
            var filters = 30;
            var filterWidth = 5;
            var filterHeight = 5;
            var poolWidth = 2;
            var poolHeight = 2;

            using (Stream
                imagesStream = assembly.GetManifestResourceStream("MNISTTest.train-images.idx3-ubyte"),
                labelsStream = assembly.GetManifestResourceStream("MNISTTest.train-labels.idx1-ubyte"))
            {
                foreach (var image in MnistImage.Load(imagesStream, labelsStream).Take(1000))
                {
                    var t = new double[10];

                    for (int i = 0; i < 10; i++)
                    {
                        if (i == image.Label)
                        {
                            t[i] = 1.0;
                        }
                        else
                        {
                            t[i] = 0.0;
                        }
                    }

                    trainingList.Add(Tuple.Create<double[], double[]>(image.Normalize(), t));
                }
            }

            using (Stream
                imagesStream = assembly.GetManifestResourceStream("MNISTTest.t10k-images.idx3-ubyte"),
                labelsStream = assembly.GetManifestResourceStream("MNISTTest.t10k-labels.idx1-ubyte"))
            {
                foreach (var image in MnistImage.Load(imagesStream, labelsStream).Take(1000))
                {
                    var t = new double[10];

                    for (int i = 0; i < 10; i++)
                    {
                        if (i == image.Label)
                        {
                            t[i] = 1.0;
                        }
                        else
                        {
                            t[i] = 0.0;
                        }
                    }

                    testList.Add(Tuple.Create<double[], double[]>(image.Normalize(), t));
                }
            }

            var inputLayer = new ConvolutionalPooling(channels, imageWidth, imageHeight, filters, filterWidth, filterHeight, poolWidth, poolHeight, new ReLU(), (index, fanIn, fanOut) => Initializers.HeNormal(fanIn));
            var hiddenLayer = new FullyConnected(inputLayer, 100, new ReLU(), (index, fanIn, fanOut) => Initializers.HeNormal(fanIn));
            var outputLayer = new Softmax(hiddenLayer, 10, (index, fanIn, fanOut) => Initializers.GlorotNormal(fanIn, fanOut));
            var network = new Network(outputLayer, new Adam(), new SoftmaxCrossEntropy());
            int epochs = 50;
            int iterations = 1;

            network.Stepped += (sender, e) =>
            {
                double tptn = 0.0;

                trainingList.ForEach(x =>
                {
                    var vector = network.Predicate(x.Item1);
                    var i = ArgMax(vector);
                    var j = ArgMax(x.Item2);

                    if (i == j && Math.Round(vector[i]) == x.Item2[j])
                    {
                        tptn += 1.0;
                    }
                });

                var accuracy = tptn / trainingList.Count;

                accuracyList.Add(accuracy);
                lossList.Add(network.Loss);

                Console.WriteLine("Epoch {0}/{1}", iterations, epochs);
                Console.WriteLine("Accuracy: {0}, Loss: {1}", accuracy, network.Loss);

                iterations++;
            };

            Console.WriteLine("Training...");

            var stopwatch = Stopwatch.StartNew();

            network.Fit(trainingList, epochs, 100);

            stopwatch.Stop();

            Console.WriteLine("Done ({0}).", stopwatch.Elapsed.ToString());

            double testTptn = 0.0;

            testList.ForEach(x =>
            {
                var vector = network.Predicate(x.Item1);
                var i = ArgMax(vector);
                var j = ArgMax(x.Item2);

                if (i == j && Math.Round(vector[i]) == x.Item2[j])
                {
                    testTptn += 1.0;
                }
            });

            Console.WriteLine("Accuracy: {0}", testTptn / testList.Count);

            logDictionary.Add("Accuracy", accuracyList);
            logDictionary.Add("Loss", lossList);

            ToCsv(logPath, logDictionary);

            Console.Write("Saved log to {0}...", logPath);
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
                s.Write(Environment.NewLine);

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
                    s.Write(Environment.NewLine);
                }

                fs.Flush();
            }
        }
    }
}
