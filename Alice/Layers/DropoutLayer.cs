using System;
using System.Collections.Generic;
using System.Text;

namespace Alice
{
    namespace Layers
    {
        public class DropoutLayer
        {
            private double dropoutProbability = 1.0;

            public double DropoutProbability
            {
                get
                {
                    return this.dropoutProbability;
                }
                set
                {
                    this.dropoutProbability = value;
                }
            }
        }
    }
}
