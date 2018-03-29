using System;

namespace Alice
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
