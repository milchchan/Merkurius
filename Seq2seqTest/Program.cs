using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using Merkurius;
using Merkurius.Layers;
using Merkurius.LossFunctions;
using Merkurius.Optimizers;

namespace Seq2seqTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Seq2seq Test");

            int seed;

            using (var rng = RandomNumberGenerator.Create())
            {
                var buffer = new byte[sizeof(int)];

                rng.GetBytes(buffer);
                seed = BitConverter.ToInt32(buffer, 0);
            }

            RandomProvider.SetSeed(seed);

            var dataList = new List<ValueTuple<int, string?, double[]>>();
            using (var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("LangModel.Wikipedia.zip"))
            using (var zipArchive = new ZipArchive(stream!))
            {
                foreach (var zipArchiveEntry in from zipArchiveEntry in zipArchive.Entries where zipArchiveEntry.FullName.EndsWith(".tsv", StringComparison.OrdinalIgnoreCase) select zipArchiveEntry)
                {
                    using (var s = zipArchiveEntry.Open())
                    using (var sr = new StreamReader(s, Encoding.UTF8))
                    {
                        var tabSeparator = new char[] { '\t' };
                        var spaceSeparator = new char[] { ' ' };
                        var isVectorStarted = false;
                        int tokenIndex = -1;
                        string? tokenName = null;
                        var weightList = new List<double>();
                        var line = sr.ReadLine();

                        while (line != null)
                        {
                            if (isVectorStarted)
                            {
                                if (line.EndsWith(']'))
                                {
                                    foreach (var weight in line.Substring(0, line.Length - 1).Split(spaceSeparator))
                                    {
                                        if (weight.Length > 0)
                                        {
                                            weightList.Add(Double.Parse(weight));
                                        }
                                    }

                                    dataList.Add(ValueTuple.Create<int, string?, double[]>(tokenIndex, tokenName, weightList.ToArray()));
                                    weightList.Clear();

                                    isVectorStarted = false;
                                }
                                else
                                {
                                    foreach (var weight in line.Split(spaceSeparator))
                                    {
                                        if (weight.Length > 0)
                                        {
                                            weightList.Add(Double.Parse(weight));
                                        }
                                    }
                                }
                            }
                            else
                            {
                                var columns = line.Split(tabSeparator);

                                tokenIndex = Int32.Parse(columns[0]);
                                tokenName = columns[1];

                                foreach (var weight in columns[2].Substring(1, columns[2].Length - 1).Split(spaceSeparator))
                                {
                                    if (weight.Length > 0)
                                    {
                                        weightList.Add(Double.Parse(weight));
                                    }
                                }

                                isVectorStarted = true;
                            }

                            line = sr.ReadLine();
                        }
                    }
                }
            }

            var wordData = new ValueTuple<string, double[]>[dataList.Count];
            var weights = new double[wordData.Length * dataList[0].Item3.Length];

            dataList.ForEach(x =>
            {
                wordData[x.Item1] = ValueTuple.Create<string, double[]>(x.Item2!, x.Item3);

                for (int i = 0, offset = x.Item1 * x.Item3.Length; i < x.Item3.Length; i++)
                {
                    weights[offset + i] = x.Item3[i];
                }
            });

            var trainingDataList = new List<ValueTuple<double[], double[]>>();
            int epochs = 100;
            int iterations = 1;
            var model = new Model(new Seq2seq(100, 100, 100, 256));

            model.MaxGradient = 5.0;
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

            model.Fit(trainingDataList, epochs, 10, new SGD(), new SoftmaxCrossEntropy());

            stopwatch.Stop();

            Console.WriteLine("Done ({0}).", stopwatch.Elapsed.ToString());
        }
    }
}
