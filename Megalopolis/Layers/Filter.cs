using System;

namespace Megalopolis
{
    namespace Layers
    {
        public interface IFilter
        {
            Batch<double[]> Forward(Batch<double[]> batch, bool isTraining);
            Batch<double[]> Backward(Batch<double[]> batch);
        }
    }
}
