using System;

namespace Merkurius
{
    public static class Initializers
    {
        public static double LeCunNormal(int fanIn)
        {
            return RandomProvider.GetRandom().Uniform(-1, 1) * Math.Sqrt(1.0 / fanIn);
        }

        public static double LeCunUniform(int fanIn)
        {
            var a = Math.Sqrt(3.0 / fanIn);

            return RandomProvider.GetRandom().Uniform(-a, a);
        }
        
        public static double GlorotNormal(int fanIn, int fanOut)
        {
            // Also known as Xavier initialization.
            return RandomProvider.GetRandom().Uniform(-1, 1) * Math.Sqrt(2.0 / (fanIn + fanOut));
        }

        public static double GlorotUniform(int fanIn, int fanOut)
        {
            // Also known as Xavier initialization.
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
