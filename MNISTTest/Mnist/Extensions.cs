using System;
using System.IO;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Mnist
{
    public static class Extensions
    {
        public static BitmapSource ToBitmapSource(this MnistImage image)
        {
            var pixels = new byte[image.Width * image.Height * 4];

            for (int i = 0, j = 0; i < pixels.Length; i+= 4)
            {
                int color = Byte.MaxValue - image.Pixels[j];

                pixels[i] = (byte)color;
                pixels[i + 1] = (byte)color;
                pixels[i + 2] = (byte)color;
                pixels[i + 3] = Byte.MaxValue;
                j++;
            }

            return BitmapSource.Create(image.Width, image.Height, 96, 96, PixelFormats.Pbgra32, null, pixels, (image.Width * PixelFormats.Pbgra32.BitsPerPixel + 7) / 8);
        }

        public static void SavePng(this BitmapSource bitmapSource, string path)
        {
            using (var stream = new FileStream(path, FileMode.Create))
            {
                var encoder = new PngBitmapEncoder();

                encoder.Frames.Add(BitmapFrame.Create(bitmapSource));
                encoder.Save(stream);
            }
        }
    }
}
