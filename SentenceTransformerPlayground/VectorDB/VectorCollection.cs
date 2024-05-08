using System;
using System.Collections.Generic;
using System.Linq;

namespace VectorDB
{
    public class VectorCollection<T> where T : IVectorObject
    {
        public VectorCollection()
        {
            Dimensions = 0;
            Objects = [];
        }

        public VectorCollection(int dimensions, List<T>? objects = null)
        {
            Dimensions = dimensions;
            Objects = objects ?? [];
        }

        public int VectorCollectionId { get; set; }

        public int Dimensions { get; set; }

        public List<T> Objects { get; set; }

        public T GetItem(int index)
        {
            return Objects[index];
        }

        public int[] CalculateRanking(float[] searchVector)
        {
            float[] scores = new float[Dimensions];
            int[] indexranks = new int[Dimensions];

            for (int i = 0; i < Dimensions; i++)
            {
                var score = CosineSimilarity(Objects[i].Vectors, searchVector);
                scores[i] = score;
            }

            var indexedFloats = scores.Select((value, index) => new { Value = value, Index = index })
              .ToArray();

            // Sort the indexed floats by value in descending order
            Array.Sort(indexedFloats, (a, b) => b.Value.CompareTo(a.Value));

            // Extract the top k indices
            indexranks = indexedFloats.Select(item => item.Index).ToArray();

            return indexranks;
        }

        private static float CosineSimilarity(float[] v1, float[] v2)
        {
            if (v1.Length != v2.Length)
            {
                throw new ArgumentException("Vectors must have the same length.");
            }
            //int size = v1.Length;
            //float m1 = Magnitude(v1);
            //float m2 = Magnitude(v2);
            /*                        var normalizedList1 = raw1.Select(o => o / m1).ToArray();
                                    var normalizedList2 = raw2.Select(o => o / m2).ToArray();
            */
            /*// Vectors should already be normalized.
            if (Math.Abs(m1 - m2) > 0.4f || Math.Abs(m1 - 1.0f) > 0.4f)
            {
                throw new InvalidOperationException("Vectors are not normalized.");
            }*/
            return DotProduct(v1, v2);
        }

        private static float CheckOverflow(double x)
        {
            if (x >= double.MaxValue)
            {
                throw new OverflowException("operation caused overflow");
            }
            return (float)x;
        }

        private static float DotProduct(float[] a, float[] b)
        {
            float result = 0.0f;
            for (int i = 0; i < a.Length; i++)
            {
                result = CheckOverflow(result + CheckOverflow(a[i] * b[i]));
            }
            return result;
        }
    }
}
