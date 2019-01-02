using System;
using System.Linq;

namespace Megalopolis
{
    namespace Layers
    {
        public class BatchNormalization : Layer, IUpdatable
        {
            private double[] weights = null;
            private double momentum = 0.9;
            private double[] means = null;
            private double[] variances = null;
            private double[] standardDeviations = null;
            private double[,] xc = null;
            private double[,] xn = null;
            private double[] dbetaVector = null;
            private double[] dgammaVector = null;

            public double[] Weights
            {
                get
                {
                    return this.weights;
                }
                set
                {
                    this.weights = value;
                }
            }

            public BatchNormalization(int nodes, Func<int, int, int, double> func) : base(nodes, nodes)
            {
                this.weights = new double[nodes * 2];
                this.means = new double[nodes];
                this.variances = new double[nodes];

                for (int i = 0, j = nodes; i < nodes; i++, j++)
                {
                    this.weights[i] = 1.0;
                    this.weights[j] = 0.0;
                    this.means[i] = 0.0;
                    this.variances[i] = 0.0;
                }
            }

            public BatchNormalization(int nodes, Func<int, int, int, double> func, double momentum) : base(nodes, nodes)
            {
                this.weights = new double[nodes * 2];
                this.means = new double[nodes];
                this.variances = new double[nodes];
                this.momentum = momentum;

                for (int i = 0, j = nodes; i < nodes; i++, j++)
                {
                    this.weights[i] = 1.0;
                    this.weights[j] = 0.0;
                    this.means[i] = 0.0;
                    this.variances[i] = 0.0;
                }
            }

            public BatchNormalization(Layer layer, Func<int, int, int, double> func) : base(layer, layer.Outputs)
            {
                this.weights = new double[layer.Outputs * 2];
                this.means = new double[layer.Outputs];
                this.variances = new double[layer.Outputs];

                for (int i = 0, j = layer.Outputs; i < layer.Outputs; i++, j++)
                {
                    this.weights[i] = 1.0;
                    this.weights[j] = 0.0;
                    this.means[i] = 0.0;
                    this.variances[i] = 0.0;
                }
            }

            public BatchNormalization(Layer layer, Func<int, int, int, double> func, double momentum) : base(layer, layer.Outputs)
            {
                this.weights = new double[layer.Outputs * 2];
                this.means = new double[layer.Outputs];
                this.variances = new double[layer.Outputs];
                this.momentum = momentum;

                for (int i = 0, j = layer.Outputs; i < layer.Outputs; i++, j++)
                {
                    this.weights[i] = 1.0;
                    this.weights[j] = 0.0;
                    this.means[i] = 0.0;
                    this.variances[i] = 0.0;
                }
            }

            public BatchNormalization(Func<int, int, int, double> func, Layer layer) : base(layer.Inputs, layer)
            {
                this.weights = new double[layer.Inputs * 2];
                this.means = new double[layer.Inputs];
                this.variances = new double[layer.Inputs];

                for (int i = 0, j = layer.Inputs; i < layer.Inputs; i++, j++)
                {
                    this.weights[i] = 1.0;
                    this.weights[j] = 0.0;
                    this.means[i] = 0.0;
                    this.variances[i] = 0.0;
                }
            }

            public BatchNormalization(Func<int, int, int, double> func, double momentum, Layer layer) : base(layer.Inputs, layer)
            {
                this.weights = new double[layer.Inputs * 2];
                this.means = new double[layer.Inputs];
                this.variances = new double[layer.Inputs];
                this.momentum = momentum;

                for (int i = 0, j = layer.Inputs; i < layer.Inputs; i++, j++)
                {
                    this.weights[i] = 1.0;
                    this.weights[j] = 0.0;
                    this.means[i] = 0.0;
                    this.variances[i] = 0.0;
                }
            }

            public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
            {
                var outputs = new Batch<double[]>(new double[inputs.Size][]);

                this.xc = new double[inputs.Size, inputs[0].Length];
                this.xn = new double[inputs.Size, inputs[0].Length];

                if (isTraining)
                {
                    var meanVector = new double[inputs[0].Length];
                    var varianceVector = new double[inputs[0].Length];
                    
                    this.standardDeviations = new double[inputs[0].Length];

                    for (int i = 0; i < meanVector.Length; i++)
                    {
                        meanVector[i] = 0.0;
                        varianceVector[i] = 0.0;
                    }

                    for (int i = 0; i < meanVector.Length; i++)
                    {
                        for (int j = 0; j < inputs.Size; j++)
                        {
                            meanVector[i] += inputs[j][i];
                        }
                    }

                    for (int i = 0; i < meanVector.Length; i++)
                    {
                        meanVector[i] = meanVector[i] / inputs.Size;
                        this.means[i] = this.momentum * this.means[i] + (1 - this.momentum) * meanVector[i];
                    }

                    for (int i = 0; i < meanVector.Length; i++)
                    {
                        for (int j = 0; j < inputs.Size; j++)
                        {
                            this.xc[j, i] = inputs[j][i] - meanVector[i];
                            varianceVector[i] += this.xc[j, i] * this.xc[j, i];
                        }
                    }

                    for (int i = 0; i < varianceVector.Length; i++)
                    {
                        varianceVector[i] = varianceVector[i] / inputs.Size;
                        this.standardDeviations[i] = Math.Sqrt(varianceVector[i] + 10e-7);
                        this.variances[i] = this.momentum * this.variances[i] + (1 - this.momentum) * varianceVector[i];
                    }

                    for (int i = 0; i < inputs[0].Length; i++)
                    {
                        for (int j = 0; j < inputs.Size; j++)
                        {
                            this.xn[j, i] = this.xc[j, i] / this.standardDeviations[i];
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < inputs[0].Length; i++)
                    {
                        for (int j = 0; j < inputs.Size; j++)
                        {
                            this.xc[j, i] = inputs[j][i] - this.means[i];
                            this.xn[j, i] = this.xc[j, i] / Math.Sqrt(this.variances[i] + 10e-7);
                        }
                    }
                }

                for (int i = 0; i < inputs.Size; i++)
                {
                    outputs[i] = new double[this.outputs];

                    for (int j = 0, k = this.outputs; j < this.outputs; j++, k++)
                    {
                        outputs[i][j] = this.weights[j] * this.xc[i, j] + this.weights[k];
                    }
                }

                return outputs;
            }

            public override Batch<double[]> Backward(Batch<double[]> deltas)
            {
                var dxn = new double[deltas.Size, deltas[0].Length];
                var dxc = new double[deltas.Size, deltas[0].Length];
                var dstd = new double[deltas[0].Length];
                var dvar = new double[deltas[0].Length];
                var dx = new Batch<double[]>(new double[deltas.Size][]);

                this.dbetaVector = new double[deltas[0].Length];
                this.dgammaVector = new double[deltas[0].Length];

                for (int i = 0; i < deltas[0].Length; i++)
                {
                    this.dbetaVector[i] = 0.0;
                    this.dgammaVector[i] = 0.0;
                    dstd[i] = 0.0;
                    dvar[i] = 0.0;
                }

                for (int i = 0; i < deltas[0].Length; i++)
                {
                    double sum = 0.0;

                    for (int j = 0; j < deltas.Size; j++)
                    {
                        this.dbetaVector[i] += deltas[j][i];
                        this.dgammaVector[i] += this.xn[j, i] * deltas[j][i];
                        dxn[j, i] = this.weights[i] * deltas[j][i];
                        dxc[j, i] = dxn[j, i] / this.standardDeviations[i];
                        dstd[i] -= dxn[j, i] * this.xc[j, i] / (this.standardDeviations[i] * this.standardDeviations[i]);
                    }

                    dvar[i] = 0.5 * dstd[i] / this.standardDeviations[i];

                    for (int j = 0; j < deltas.Size; j++)
                    {
                        dxc[j, i] += (2.0 / deltas.Size) * this.xc[j, i] * dvar[i];
                        sum += dxc[j, i];
                    }

                    for (int j = 0; j < deltas.Size; j++)
                    {
                        dx[j][i] = dxc[j, i] - sum / deltas.Size;
                    }
                }

                return dx;
            }

            public Batch<double[]> GetGradients()
            {
                return new Batch<double[]>(new double[1][] { this.dgammaVector.Concat<double>(this.dbetaVector).ToArray<double>() });
            }

            public void Update(Batch<double[]> gradients, Func<double, double, double> func)
            {
                foreach (var vector in gradients)
                {
                    for (int i = 0, j = this.outputs; i < vector.Length; i++, j++)
                    {
                        this.weights[i] = func(this.weights[i], vector[i]);
                        this.weights[j] = func(this.weights[j], vector[j]);
                    }
                }
            }
        }
    }
}
