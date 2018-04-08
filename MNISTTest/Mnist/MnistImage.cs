using System;
using System.Collections.Generic;
using System.IO;

namespace Mnist
{
    public class MnistImage
    {
        // an MNIST image of a '0' thru '9' digit
        private int width; // 28
        private int height; // 28
        private byte[] pixels; // 0(white) - 255(black)
        private byte label; // '0' - '9'

        public int Width
        {
            get
            {
                return this.width;
            }
        }

        public int Height
        {
            get
            {
                return this.height;
            }
        }

        public byte[] Pixels
        {
            get
            {
                return this.pixels;
            }
        }

        public byte Label
        {
            get
            {
                return this.label;
            }
        }

        public MnistImage(int width, int height, byte[] pixels, byte label)
        {
            var length = height * width;

            this.width = width;
            this.height = height;
            this.pixels = new byte[length];

            for (int i = 0; i < length; ++i)
            {
                this.pixels[i] = pixels[i];
            }

            this.label = label;
        }

        public double[] Normalize()
        {
            var length = this.height * this.width;
            var normalizeDpixels = new double[length];

            for (int i = 0; i < length; ++i)
            {
                normalizeDpixels[i] = (double)this.pixels[i] / Byte.MaxValue;
            }

            return normalizeDpixels;
        }

        public static IEnumerable<MnistImage> Load(Stream imagesStream, Stream labelsStream)
        {
            // Load MNIST dataset
            const int maxImageWidth = 28;
            const int maxImageHeight = 28;
            var maxPixels = maxImageHeight * maxImageWidth;
            byte[] pixels = new byte[maxPixels];
            List<MnistImage> imageList = new List<MnistImage>();

            using (BinaryReader imagesBinaryReader = new BinaryReader(imagesStream), labelsBinaryReader = new BinaryReader(labelsStream))
            {
                long maxImages = (imagesBinaryReader.BaseStream.Length - 4 * 4) / (28 * 28);
                int magic1 = imagesBinaryReader.ReadInt32(); // stored as Big Endian

                magic1 = ReverseBytes(magic1); // convert to Intel format

                int imageCount = imagesBinaryReader.ReadInt32();

                imageCount = ReverseBytes(imageCount);

                int numRows = imagesBinaryReader.ReadInt32();

                numRows = ReverseBytes(numRows);

                int numCols = imagesBinaryReader.ReadInt32();

                numCols = ReverseBytes(numCols);

                int magic2 = labelsBinaryReader.ReadInt32();

                magic2 = ReverseBytes(magic2);

                int numLabels = labelsBinaryReader.ReadInt32();

                numLabels = ReverseBytes(numLabels);

                for (int i = 0; i < maxImages; ++i)
                {
                    for (int j = 0; j < maxPixels; ++j)
                    {
                        pixels[j] = imagesBinaryReader.ReadByte();
                    }

                    imageList.Add(new MnistImage(maxImageWidth, maxImageHeight, pixels, labelsBinaryReader.ReadByte()));
                }
            }

            return imageList;
        }

        private static int ReverseBytes(int v)
        {
            // bit-manipulation version
            //  return (v & 0x000000FF) << 24 | (v & 0x0000FF00) << 8 |
            //         (v & 0x00FF0000) >> 8 | ((int)(v & 0xFF000000)) >> 24;
            byte[] intAsBytes = BitConverter.GetBytes(v);

            Array.Reverse(intAsBytes);

            return BitConverter.ToInt32(intAsBytes, 0);
        }
    }
}
