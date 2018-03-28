using System;
using System.Collections.Generic;
using System.Text;

namespace Alice
{
    namespace Layers
    {
        public class MaxPoolingLayer
        {
            private double[] inputActivations = null;
            private double[] outputActivations = null;
            private int width = 0;
            private int height = 0;
            private int stride = 1;
            private int padding = 0;

            public MaxPoolingLayer(int width, int height)
            {
                this.width = width;
                this.height = height;
            }

            public MaxPoolingLayer(int width, int height, int stride, int padding)
            {
                this.width = width;
                this.height = height;
                this.stride = stride;
                this.padding = padding;
            }

            public void Forward()
            {

            }

            public void Backward()
            {

            }


        //    minibatch_size = len(input)


            //s0 = len(input[0][0][0]) / self.pool_size[0]
            //s1 = len(input[0][0][1]) / self.pool_size[1]


            //pooled_input = numpy.zeros((len(input), self.n_kernel, s0, s1) )


            //for batch in xrange(minibatch_size) :
            //    for k in xrange(self.n_kernel) :

            //        for i in xrange(s0) :
            //            for j in xrange(s1) :

            //                for s in xrange(self.pool_size[0]) :
            //                    for t in xrange(self.pool_size[1]) :

            //                        if s == 0 and t == 0:
            //                            max_ = input[batch][k][self.pool_size[0] * i][self.pool_size[1] * j]
            //                            next

            //                        if max_<input[batch][k][self.pool_size[0] * i + s][self.pool_size[1] * j + t]:
            //                            max_ = input[batch][k][self.pool_size[0] * i + s][self.pool_size[1] * j + t]


            //                pooled_input[batch][k][i][j] = max_

            //self.pooled_input = pooled_input
            //return pooled_input
        }
    }
}
