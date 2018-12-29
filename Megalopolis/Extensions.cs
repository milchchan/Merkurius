using System;
using System.Collections.Generic;
using System.Linq;

namespace Megalopolis
{
    public static class Extensions
    {
        public static double Uniform(this Random random, double min, double max)
        {
            return (max - min) * random.NextDouble() + min;
        }

        public static IEnumerable<T> Sample<T>(this IEnumerable<T> collection, Random random, int size)
        {
            // Generates a random sample from a given 1-D collection
            T[] array1 = collection.ToArray();
            T[] array2 = new T[array1.Length < size ? array1.Length : size];
            int max = collection.Count();
            int i = 0;

            while (i < array2.Length)
            {
                int j = random.Next(i, max);
                T temp = array1[i];

                array1[i] = array2[i] = array1[j];
                array1[j] = temp;

                i++;
            }

            return array2;
        }
    }
}
