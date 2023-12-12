using System;
using System.Threading;

namespace Merkurius
{
    public static class RandomProvider
    {
        private static int seed = Environment.TickCount;
        [ThreadStatic]
        private static Random? random = null;

        public static void SetSeed(int seed)
        {
            RandomProvider.seed = seed;
            RandomProvider.random = null;
        }

        public static System.Random GetRandom()
        {
            return RandomProvider.random ?? (RandomProvider.random = new Random(Interlocked.Increment(ref RandomProvider.seed)));
        }
    }
}
