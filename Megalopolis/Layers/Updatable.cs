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
            Batch<double[]> GetGradients();
            void SetGradients(Func<bool, double, int, double> func);
            void Update(Batch<double[]> gradients, Func<double, double, double> func);
        }
    }
}
