using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Megalopolis
{
    namespace Layers
    {
        public class Embedding : Layer
        {
            public Embedding(int inputs, int outputs) : base(inputs, outputs) { }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                throw new NotImplementedException();
            }

            public override Tuple<Batch<double[]>, Batch<double[]>> Backward(Batch<double[]> inputs, Batch<double[]> outputs, Batch<double[]> deltas)
            {
                throw new NotImplementedException();
            }

            public override void Update(Batch<double[]> gradients, Func<double, double, double> func) { }
        }
    }
}
