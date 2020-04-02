using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using Merkurius;
using Merkurius.Layers;
using Merkurius.LossFunctions;
using Merkurius.Optimizers;

namespace SineWaveTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Sine Wave Test");
            int seed;

            using (var rng = new RNGCryptoServiceProvider())
            {
                var buffer = new byte[sizeof(int)];

                rng.GetBytes(buffer);
                seed = BitConverter.ToInt32(buffer, 0);
            }

            RandomProvider.SetSeed(seed);

            var trainingDataList = new List<Tuple<double[], double[]>>();
            var dataSize = 100;
            var maxLength = 200;

            for (var i = 0; i < dataSize; i++)
            {
                var x = new double[maxLength];
                var y = new double[maxLength];

                for (var j = 0; j < maxLength; j++)
                {
                    x[j] = Math.Sin((i + j) * 0.01 * Math.PI);
                    y[j] = Math.Sin((i + j + maxLength) * 0.01 * Math.PI);
                }

                trainingDataList.Add(Tuple.Create<double[], double[]>(x, y));
            }

            int epochs = 100;
            int iterations = 1;
            Model model = new Model(
                new Recurrent(1, 128, maxLength, true, (fanIn, fanOut) => RandomProvider.GetRandom().NextDouble(),
                new FullyConnected(128, maxLength, 1, (fanIn, fanOut) => RandomProvider.GetRandom().NextDouble())),
                new SGD(), new MeanSquaredError());

            model.Stepped += (sender, args) =>
            {
                if (iterations % 10 == 0)
                {
                    Console.WriteLine("Epoch {0}/{1}", iterations, epochs);
                }

                iterations++;
            };

            Console.WriteLine("Training...");

            var stopwatch = Stopwatch.StartNew();

            model.Fit(trainingDataList, epochs, 1);

            stopwatch.Stop();

            Console.WriteLine("Done ({0}).", stopwatch.Elapsed.ToString());

            var dataList = new List<IEnumerable<double>>();
            var data = trainingDataList[0];
            var vector = model.Predicate(data.Item1);

            for (var i = 0; i < data.Item1.Length; i++)
            {
                dataList.Add(new double[] { data.Item1[i], data.Item2[i], vector[i] });
            }

            using (var fs = new FileStream("SineWaveTest.csv", FileMode.Create, FileAccess.Write, FileShare.Read))
            using (var s = new StreamWriter(fs, System.Text.Encoding.UTF8))
            {
                dataList.ForEach(x =>
                {
                    s.Write(String.Join(",", x));
                    s.Write(Environment.NewLine);
                });

                fs.Flush();
            }
        }
    }
}
