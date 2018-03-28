using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Alice.Layers;

namespace Alice
{
    public interface ITrainer
    {
        void Train(Collection<Layer> layerList, IDictionary<double[], IEnumerable<double[]>> dictionary, int epochs);
    }
}
