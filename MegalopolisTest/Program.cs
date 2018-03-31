using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Diagnostics;
using Alice;
using Alice.ActivationFunctions;
using Alice.Layers;
using Alice.LossFunctions;
using Alice.Optimizers;


namespace MegalopolisTest
{
    class Program
    {
        static void Main(string[] args)
        {
            int seed = Environment.TickCount;
            Random random = new Random(seed);
            Dictionary<double[], IEnumerable<double[]>> patternDictionary = new Dictionary<double[], IEnumerable<double[]>>();
            /*Network autoencoder = new Network(random, new FullyConnectedLayer[] {
                new FullyConnectedLayer(3, 2, new HyperbolicTangent()),
                new FullyConnectedLayer(1, new HyperbolicTangent())
            }, (x, y) => -Math.Sqrt(6 / (x + y)), (x, y) => Math.Sqrt(6 / (x + y)), new StackedDenoisingAutoencoder(random));*/
            //Model model = new Model(random, new Layer[] { new Layer(3, new HyperbolicTangent()), new Layer(2, new HyperbolicTangent()), new Layer(1, new HyperbolicTangent()) }, (x, y) => -Math.Sqrt(6 / (x + y)), (x, y) => Math.Sqrt(6 / (x + y)), new Backpropagation(random, new AdaDelta(), new MeanSquaredError()) { ErrorThreshold = 0.001 });
            //Model model = new Model(random, new Layer[] { new Layer(3, new Sigmoid()), new Layer(2, new Sigmoid()), new Layer(1, new Sigmoid()) }, (x, y) => -Math.Sqrt(6 / (x + y)) * 4, (x, y) => Math.Sqrt(6 / (x + y)) * 4, new Backpropagation(random, new Momentum(0.5, 0.1), new MeanSquaredError()) { ErrorThreshold = 0.001 }); // For Sigmoid activation function

            patternDictionary.Add(new double[] { 0 }, new double[][] { new double[] { 0, 0 } });
            patternDictionary.Add(new double[] { 1 }, new double[][] { new double[] { 0, 1 } });
            patternDictionary.Add(new double[] { 1 }, new double[][] { new double[] { 1, 0 } });
            patternDictionary.Add(new double[] { 0 }, new double[][] { new double[] { 1, 1 } });

            /*patternDictionary.Add(new double[] { 0, 1 }, new double[][] { new double[] { 0, 0 } });
            patternDictionary.Add(new double[] { 1, 0 }, new double[][] { new double[] { 0, 1 } });
            patternDictionary.Add(new double[] { 1, 0 }, new double[][] { new double[] { 1, 0 } });
            patternDictionary.Add(new double[] { 0, 1 }, new double[][] { new double[] { 1, 1 } });*/

            /*foreach (Layer layer in model.Layers)
            {
                layer.DropoutProbability = 0.5;
            }*/

            Console.WriteLine("XOR Test ({0})", seed);

            /*Console.Write("Pretraining...");*/

            Stopwatch sw = Stopwatch.StartNew();

            /*autoencoder.Train(patternDictionary, 1000);

            sw.Stop();

            Console.WriteLine("Done ({0}).", sw.Elapsed.ToString());*/

            /*List<FullyConnectedLayer> layerList = new List<FullyConnectedLayer>();
            List<double> weightList = new List<double>();

            foreach (FullyConnectedLayer layer in autoencoder.Layers)
            {
                FullyConnectedLayer copiedLayer = new FullyConnectedLayer(layer.Activations.Length, new HyperbolicTangent());

                for (int i = 0; i < layer.Activations.Length; i++)
                {
                    copiedLayer.Activations[i] = layer.Activations[i];
                }

                layerList.Add(copiedLayer);
            }

            foreach (double[,] weights in autoencoder.Weights)
            {
                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    for (int j = 0; j < weights.GetLength(1); j++)
                    {
                        weightList.Add(weights[i, j]);
                    }
                }
            }*/

            //Network backpropagation = new Network(layerList, (i) => weightList[i], new Backpropagation(random, new AdaDelta(), new MeanSquaredError()) { ErrorThreshold = 0.001 });
            Backpropagation backpropagation = new Backpropagation(random, new AdaDelta(), new MeanSquaredError());
            Network network = new Network(random, new FullyConnectedLayer[] {
                new FullyConnectedLayer(2, new Sigmoid()),
                new FullyConnectedLayer(2, new Sigmoid()),
                new FullyConnectedLayer(1, new Sigmoid())
            }, (x, y) => -Math.Sqrt(6 / (x + y)) * 4, (x, y) => Math.Sqrt(6 / (x + y)) * 4, backpropagation);

            sw.Reset();

            Console.Write("Training...");

            sw.Start();

            network.Train(patternDictionary, 10000000);

            sw.Stop();

            Console.WriteLine("Done ({0}).", sw.Elapsed.ToString());
            Console.WriteLine();

            foreach (double[] vector in patternDictionary.Values.Aggregate<IEnumerable<double[]>, List<double[]>>(new List<double[]>(), (list, vectors) =>
            {
                foreach (double[] vector in vectors)
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
            Console.WriteLine("Loss: {0}", backpropagation.Loss);
            Console.WriteLine();
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
