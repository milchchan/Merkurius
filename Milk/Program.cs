using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using Milk.ActivationFunctions;
using Milk.Optimizers;

namespace Milk
{
    class Program
    {
        static void Main(string[] args)
        {
            Dictionary<double[], double[][]> patternDictionary = new Dictionary<double[], double[][]>();
            Model model = new Model(Environment.TickCount, 2, 2, 1, 1, (x, y) => -Math.Sqrt(6 / (x + y)), (x, y) => Math.Sqrt(6 / (x + y)), new HyperbolicTangent(), new AdaDelta());
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

            model.Pretrain(patternDictionary, 0.1, 0.3, 1000);

            sw.Stop();

            Console.WriteLine("Done ({0}).", sw.Elapsed.ToString());

            sw.Reset();

            Console.Write("Training...");

            sw.Start();

            model.Train(patternDictionary, 1000000);

            sw.Stop();

            Console.WriteLine("Done ({0}).", sw.Elapsed.ToString());
            Console.WriteLine();

            foreach (double[] vector in patternDictionary.Values.Aggregate<double[][], List<double[]>>(new List<double[]>(), (list, array) =>
            {
                foreach (double[] vector in array)
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
