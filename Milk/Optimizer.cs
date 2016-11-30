using System;

namespace Milk
{
    namespace Optimizers
    {
        public interface IOptimizer
        {
            double Optimize(int index, double weight, double gradient);
        }
    }
}
