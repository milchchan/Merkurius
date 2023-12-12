using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Merkurius
{
    public class Batch<T> : IEnumerable<T>
    {
        private T[]? items = null;

        public T this[long i]
        {
            get
            {
                return this.items![i];
            }
            set
            {
                this.items![i] = value;
            }
        }

        public int Size
        {
            get
            {
                return this.items!.Length;
            }
        }

        public bool IsReadOnly
        {
            get
            {
                return false;
            }
        }

        public Batch(IEnumerable<T> collection)
        {
            this.items = collection.ToArray<T>();
        }

        public IEnumerator<T> GetEnumerator()
        {
            return this.items.Cast<T>().GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.items!.GetEnumerator();
        }
    }
}
