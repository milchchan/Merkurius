using System;

namespace Megalopolis
{
    namespace Layers
    {
        public interface IFilter
        {
            Batch<double[]> PropagateForward(Batch<double[]> batch, bool isTraining);
            Batch<double[]> PropagateBackward(Batch<double[]> batch);
        }
    }
}
