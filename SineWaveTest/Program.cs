using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Security.Cryptography;
using Merkurius;
using Merkurius.ActivationFunctions;
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
            var dataSize = 10;
            var maxLength = 20;

            for (var i = 0; i < dataSize; i++)
            {
                var x = new double[maxLength];

                for (var j = 0; j < maxLength; j++)
                {
                    x[j] = Math.Sin((i + j) * 0.1 * Math.PI);
                }

                trainingDataList.Add(Tuple.Create<double[], double[]>(x, new double[] { Math.Sin((i + maxLength) * 0.1 * Math.PI) }));
            }

            Model model;

            int epochs = 100;
            int iterations = 1;

            model = new Model(
                new Recurrent(maxLength, 128, 5, true, (fanIn, fanOut) => RandomProvider.GetRandom().NextDouble(),
                new FullyConnected(20, 1, (fanIn, fanOut) => RandomProvider.GetRandom().NextDouble())),
                new Momentum(0.5, 0.1), new MeanSquaredError());

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

            model.Fit(trainingDataList, epochs);

            stopwatch.Stop();

            Console.WriteLine("Done ({0}).", stopwatch.Elapsed.ToString());
        }
    }
}
