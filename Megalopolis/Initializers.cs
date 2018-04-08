using System;

namespace Megalopolis
{
    public static class Initializers
    {
        public static double LecunNormal(int fanIn)
        {
            return RandomProvider.GetRandom().Uniform(-1, 1) * Math.Sqrt(1.0 / fanIn);
        }

        public static double LecunUniform(int fanIn)
        {
            var a = Math.Sqrt(3.0 / fanIn);

            return RandomProvider.GetRandom().Uniform(-a, a);
        }

        public static double GlorotNormal(int fanIn, int fanOut)
        {
            return RandomProvider.GetRandom().Uniform(-1, 1) * Math.Sqrt(2.0 / (fanIn + fanOut));
        }

        public static double GlorotUniform(int fanIn, int fanOut)
        {
            var a = Math.Sqrt(6.0 / (fanIn + fanOut));

            return RandomProvider.GetRandom().Uniform(-a, a);
        }

        public static double HeNormal(int fanIn)
        {
            return RandomProvider.GetRandom().Uniform(-1, 1) * Math.Sqrt(2.0 / fanIn);
        }

        public static double HeUniform(int fanIn)
        {
            var a = Math.Sqrt(6.0 / fanIn);

            return RandomProvider.GetRandom().Uniform(-a, a);
        }
    }
}
