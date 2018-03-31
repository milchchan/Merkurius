using System;

namespace Alice
{
    namespace Optimizers
    {
        public interface IOptimizer
        {
            double Optimize(int index, double weight, double gradient);
        }
    }
}
