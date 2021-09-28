#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "ulcg.h"
// Librerías necesarias para LCGs
#include "gdef.h"
#include "unif01.h"
// Librerías necesarias para la batería de tests
#include "bbattery.h"

#define M 2147483647
#define A 16807
#define C 0
#define DEFAULT_SEED 559079219

int main (char* argv[], int argc)
{
   int seed = (argc > 1) ?  atoi(argv[1]) : DEFAULT_SEED;
   unif01_Gen *gen;
   gen = ulcg_CreateLCG (M, A, C, seed);
   bbattery_SmallCrush (gen);
   ulcg_DeleteGen (gen);

   printf("Nuestra semilla: %d\n", seed);

   return 0;
}
