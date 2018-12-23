using System;

namespace Megalopolis
{
    namespace Layers
    {
        public interface IUpdatable
        {
            double[] Weights
            {
                get;
                set;
            }
            void Update(Batch<double[]> gradients, Func<double, double, double> func);
        }
    }
}
