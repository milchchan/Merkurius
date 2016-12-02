using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using Milk.ActivationFunctions;
using Milk.LossFunctions;
using Milk.Optimizers;

namespace Milk
{
    class Program
    {
        static void Main(string[] args)
        {
            Random random = new Random(Environment.TickCount);
            Dictionary<double[], IEnumerable<double[]>> patternDictionary = new Dictionary<double[], IEnumerable<double[]>>();
            Model model = new Model(random, new Layer[] { new Layer(3, new HyperbolicTangent(), new AdaDelta()), new Layer(2, new HyperbolicTangent(), new AdaDelta()), new Layer(1, new HyperbolicTangent(), new AdaDelta()) }, (x, y) => -Math.Sqrt(6 / (x + y)), (x, y) => Math.Sqrt(6 / (x + y)), new MeanSquaredError());
            //Model model = new Model(Environment.TickCount, 2, 2, 1, 1, (x, y) => -Math.Sqrt(6 / (x + y)), (x, y) => Math.Sqrt(6 / (x + y)), new HyperbolicTangent(), new AdaDelta());
            //Model model = new Model(Environment.TickCount, 2, 2, 1, 1, (x, y) => -Math.Sqrt(6 / (x + y)) * 4, (x, y) => Math.Sqrt(6 / (x + y)) * 4, new Sigmoid(), new Momentum(0.5, 0.1)); // For Sigmoid activation function

            patternDictionary.Add(new double[] { 0 }, new double[][] { new double[] { 0, 0 } });
            patternDictionary.Add(new double[] { 1 }, new double[][] { new double[] { 0, 1 } });
            patternDictionary.Add(new double[] { 1 }, new double[][] { new double[] { 1, 0 } });
            patternDictionary.Add(new double[] { 0 }, new double[][] { new double[] { 1, 1 } });

            /*patternDictionary.Add(new double[] { 0, 1 }, new double[][] { new double[] { 0, 0 } });
            patternDictionary.Add(new double[] { 1, 0 }, new double[][] { new double[] { 0, 1 } });
            patternDictionary.Add(new double[] { 1, 0 }, new double[][] { new double[] { 1, 0 } });
            patternDictionary.Add(new double[] { 0, 1 }, new double[][] { new double[] { 1, 1 } });*/

            model.ErrorThreshold = 0.001;

            foreach (Layer layer in model.Layers)
            {
                layer.DropoutProbability = 1.0;
            }

            Console.WriteLine("XOR Test");

            Console.Write("Pretraining...");

            Stopwatch sw = Stopwatch.StartNew();

            model.Pretrain(patternDictionary.Values.Aggregate<IEnumerable<double[]>, List<double[]>>(new List<double[]>(), (list, vectors) =>
            {
                foreach (double[] vector in vectors)
                {
                    list.Add(vector);
                }

                return list;
            }), 0.1, 0.3, 1000);

            sw.Stop();

            Console.WriteLine("Done ({0}).", sw.Elapsed.ToString());

            sw.Reset();

            Console.Write("Training...");

            sw.Start();

            model.Train(patternDictionary, 1000000);

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
                })), String.Join(",", model.Predicate(vector).Aggregate<double, List<string>>(new List<string>(), (x, y) =>
                {
                    x.Add(y.ToString());

                    return x;
                })));
            }

            Console.WriteLine();
        }
    }
}
