void sample(int* x, int Nx, int b){
  int i;
  for (i=0;i<Nx;i++) {
    
    x[i] = b*i;
    printf("%d \n",i);
       
  }
  
}
