/*
 * ura.h
 *
 * ura_init()     .. inicializacija
 * ura()          .. vrne dva casa
 */

#ifndef URAdatoteka
#define URAdatoteka
#include <stdio.h>

#define URA(x)    time_cur = ura(time_start, time_cur,(x))

struct timeval ura_init()
{
  struct timeval time_start;
  gettimeofday(&time_start, (struct timezone*)0);   /* current time */
  return time_start;
}

struct timeval ura(struct timeval tv_start, struct timeval tv_previous,
               char *izpis)
{
  struct timeval tv2;
/********** Popravil 17.12.1997 *********
  long dt1;
  long dtsum;  
**************/
  double dt1, dtsum; 
  
  gettimeofday(&tv2, (struct timezone*)0);
  dtsum = ((double)(tv2.tv_sec) - (double)(tv_start.tv_sec)) * 1000000 +
        (double)(tv2.tv_usec) - (double)(tv_start.tv_usec);
  dt1 = ((double)(tv2.tv_sec) - (double)(tv_previous.tv_sec)) * 1000000 +  
          (double)(tv2.tv_usec) - (double)(tv_previous.tv_usec);

  /***
  printf("%s: %.3f [%.3f] sekund\n", izpis, 
         (double)(dt1)/1000000, (double)(dtsum)/1000000);
  ***/
  printf("%s: %.3f [%.3f] sekund\n", izpis, dt1/1000000, dtsum/1000000);
    
  return tv2;
}

/* Funkcija vrne razliko casov v SEKUNDAH */
double ura_razlika(struct timeval t_pred, struct timeval t_zdaj)
{
   /***
   long dt1;
   ***/
   double dt1;
   
   dt1 = (double)((t_zdaj.tv_sec) - (double)(t_pred.tv_sec)) * 1000000 +
          (double)(t_zdaj.tv_usec) - (double)(t_pred.tv_usec);
   return dt1/1000000;
}
#endif
