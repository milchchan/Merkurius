using System;

namespace Megalopolis
{
    namespace Optimizers
    {
        public interface IOptimizer
        {
            double Optimize(int index, double weight, double gradient);
        }
    }
}
