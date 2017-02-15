#include "defines.hpp"

void QuicksortInverse(int *pOffsets, const int *pValues, int nLow, int nHigh)
{
        int i = nLow;
        int j = nHigh;

        const int x = pValues[pOffsets[(nLow + nHigh) >> 1]];

        while (i <= j)
        {
                while (pValues[pOffsets[i]] > x) i++;
                while (pValues[pOffsets[j]] < x) j--;

                if (i <= j)
                {
                        const int temp = pOffsets[i];
                        pOffsets[i] = pOffsets[j];
                        pOffsets[j] = temp;

                        i++;
                        j--;
                }
        }

        if (nLow < j) QuicksortInverse(pOffsets, pValues, nLow, j);
        if (i < nHigh) QuicksortInverse(pOffsets, pValues, i, nHigh);
}
