using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Megalopolis.Layers;

namespace Megalopolis
{
    public interface ITrainer
    {
        void Train(Collection<Layer> layerList, IDictionary<double[], IEnumerable<double[]>> dictionary, int epochs);
    }
}
