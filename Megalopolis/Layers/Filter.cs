using System;

namespace Megalopolis
{
    namespace Layers
    {
        public interface IFilter
        {
            double[] PropagateForward(bool isTraining, double[] activations);
            double[] PropagateBackward(double[] gradients);
        }
    }
}
