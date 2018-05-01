using System;
using System.Collections;
using System.Collections.Generic;

namespace Megalopolis
{
    public class Batch<T> : ICollection<T>
    {
        private List<T> itemList = null;

        public T this[int i]
        {
            get
            {
                return this.itemList[i];
            }
            set
            {
                this.itemList[i] = value;
            }
        }

        public int Count
        {
            get
            {
                return this.itemList.Count;
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
            this.itemList = new List<T>(collection);
        }

        public void Add(T item)
        {
            this.itemList.Add(item);
        }

        public void Clear()
        {
            this.itemList.Clear();
        }

        public bool Contains(T item)
        {
            return this.itemList.Contains(item);
        }

        public void CopyTo(T[] array, int arrayIndex)
        {
            this.itemList.CopyTo(array, arrayIndex);
        }

        public bool Remove(T item)
        {
            return this.itemList.Remove(item);
        }

        public IEnumerator<T> GetEnumerator()
        {
            return this.itemList.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.itemList.GetEnumerator();
        }
    }
}
