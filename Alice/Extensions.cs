using System;
using System.Collections.Generic;
using System.Linq;

namespace Alice
{
    public static class Extensions
    {
        public static int Binomial(this Random random, int n, double p)
        {
            int count = 0;

            for (int i = 0; i < n; i++)
            {
                if (random.NextDouble() < p)
                {
                    count++;
                }
            }

            return count;
        }

        public static double Uniform(this Random random, double min, double max)
        {
            return (max - min) * random.NextDouble() + min;
        }

        public static IEnumerable<T> Shuffle<T>(this IEnumerable<T> collection, Random random)
        {
            // Fisher-Yates algorithm
            T[] array = collection.ToArray();
            int n = array.Length; // The number of items left to shuffle (loop invariant).

            while (n > 1)
            {
                int k = random.Next(n); // 0 <= k < n.

                n--; // n is now the last pertinent index;
                T temp = array[n]; // swap list[n] with list[k] (does nothing if k == n).
                array[n] = array[k];
                array[k] = temp;
            }

            return array;
        }

        public static IEnumerable<T> Sample<T>(this IEnumerable<T> collection, Random random, int size)
        {
            // Generates a random sample from a given 1-D collection
            int max = collection.Count();
            int n = size > max ? max : size;
            int i = 0;
            Dictionary<int, T> dictionary = new Dictionary<int, T>();

            while (i < n)
            {
                int j = random.Next(i, n);
                T temp;

                if (dictionary.TryGetValue(j, out temp))
                {
                    yield return temp;

                    dictionary[j] = collection.ElementAt(i);
                }
                else
                {
                    yield return collection.ElementAt(j);

                    dictionary.Add(j, collection.ElementAt(i));
                }

                i++;
            }
        }
    }
}
