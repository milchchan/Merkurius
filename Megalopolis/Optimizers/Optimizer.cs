using System;

namespace Megalopolis
{
    namespace Optimizers
    {
        public interface IOptimizer
        {
            double Optimize(int i, double w, double dw);
        }
    }
}
