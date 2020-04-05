using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Security.Cryptography;
using System.Xml;
using Merkurius;
using Merkurius.ActivationFunctions;
using Merkurius.Layers;
using Merkurius.LossFunctions;
using Merkurius.Optimizers;

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

            var filename = "XOR.xml";
            var serializer = new DataContractSerializer(typeof(IEnumerable<Layer>), new Type[] { typeof(FullyConnected), typeof(BatchNormalization), typeof(Activation), typeof(Sigmoid) });
            var patternList = new List<Tuple<double[], double[]>>();
            var accuracyList = new List<double>();
            var lossList = new List<double>();
            Model model;

            patternList.Add(Tuple.Create<double[], double[]>(new double[] { 0, 0 }, new double[] { 0 }));
            patternList.Add(Tuple.Create<double[], double[]>(new double[] { 0, 1 }, new double[] { 1 }));
            patternList.Add(Tuple.Create<double[], double[]>(new double[] { 1, 0 }, new double[] { 1 }));
            patternList.Add(Tuple.Create<double[], double[]>(new double[] { 1, 1 }, new double[] { 0 }));

            if (File.Exists(filename))
            {
                using (XmlReader xmlReader = XmlReader.Create(filename))
                {
                    model = new Model((IEnumerable<Layer>)serializer.ReadObject(xmlReader), new Momentum(0.5, 0.1), new SoftmaxCrossEntropy());
                }
            }
            else
            {
                model = new Model(
                    new FullyConnected(2, (fanIn, fanOut) => RandomProvider.GetRandom().NextDouble(),
                    new Activation(new Sigmoid(),
                    new FullyConnected(2, 1, (fanIn, fanOut) => RandomProvider.GetRandom().NextDouble()))),
                    new Momentum(0.5, 0.1), new SoftmaxCrossEntropy());

                int epochs = 10000;
                int iterations = 1;

                //model.WeightDecayRate = 0.01;
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
            }

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

            XmlWriterSettings settings = new XmlWriterSettings();

            settings.Indent = true;
            settings.Encoding = new System.Text.UTF8Encoding(false);

            using (XmlWriter xmlWriter = XmlWriter.Create(filename, settings))
            {
                serializer.WriteObject(xmlWriter, model.Layers);
                xmlWriter.Flush();
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
