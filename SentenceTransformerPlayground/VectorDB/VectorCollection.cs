using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

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

        public static float CheckOverflow(double x)
        {
            if (x >= double.MaxValue)
            {
                throw new OverflowException("operation caused overflow");
            }
            return (float)x;
        }

        public static float DotProduct(float[] a, float[] b)
        {
            float result = 0.0f;
            for (int i = 0; i < a.Length; i++)
            {
                result = CheckOverflow(result + CheckOverflow(a[i] * b[i]));
            }
            return result;
        }

        public static float Magnitude(float[] v)
        {
            float result = 0.0f;
            for (int i = 0; i < v.Length; i++)
            {
                result = CheckOverflow(result + CheckOverflow(v[i] * v[i]));
            }
            return (float)Math.Sqrt(result);
        }

        public async Task SaveToDiskAsync(string fileName, CancellationToken ct = default)
        {
            using var db = new VectorDBContext<T>(fileName);

            await db.Database.EnsureDeletedAsync(ct);

            await db.Database.EnsureCreatedAsync(ct);

            await db.AddAsync(this, ct);

            await db.SaveChangesAsync(ct);
        }

        public static async Task<VectorCollection<T>?> LoadFromDiskAsync(string fileName, CancellationToken ct = default)
        {
            using var db = new VectorDBContext<T>(fileName);

            try
            {
                return await db.VectorCollections
                    .Include(v => v.Objects).FirstAsync(ct).ConfigureAwait(false);
            }
            catch (Exception)
            {
                return null;
            }
        }
    }

    public class VectorDBContext<T>(string fileName) : DbContext where T : IVectorObject
    {
        public DbSet<VectorCollection<T>> VectorCollections { get; set; }

        public static string DbPath(string fileName)
        {
            return Path.Join(Windows.Storage.ApplicationData.Current.LocalFolder.Path, fileName);
        }

        protected override void OnConfiguring(DbContextOptionsBuilder options) => options.UseSqlite($"Data Source={DbPath(fileName)}");
    }
}
