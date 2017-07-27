#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <valarray>
using namespace std;


class OptimalControlProblem {
public:
  double T,y0;
  int N;
  double dt;

  OptimalControlProblem(double,double);
  
  void set_N(int);

  double ODE_update(int,vector<double>&,double);
  vector<double> ODE_solver(int, vector<double>&);
  
  
};

OptimalControlProblem::OptimalControlProblem(double a,double b){
  T=a;
  y0=b;
}
void OptimalControlProblem::set_N(int n){
  N=n;
  dt = T/N;
}
double OptimalControlProblem::ODE_update(int j, vector<double>& u, double y) {
  return y + dt*u[j];
}

vector<double> OptimalControlProblem::ODE_solver(int N, vector<double>& u) {
  
  vector<double> y(N+1);

  set_N(N);
  
  y[0]=y0;
  int i;
  for(i=0;i<N;i++) {
    y[i+1] = ODE_update(i,u,y[i]);
  }

  return y;
  
}

int main()
{
  int i;

  vector<double> v1(11);
  cout << "(";
  for (i=0;i<11;i++){
    v1[i] = 0.1*i;
    cout << v1[i]<<",";
  }
  cout << ")" << "\n";
  
  int N=100;
  OptimalControlProblem problem (1,1.2);
  
  
  
  cout << "hello" << "\n";

  vector<double> u(N+1);

  for(i=0;i<N+1;i++) {
    u[i]=0;
  }

  vector<double>y = problem.ODE_solver(N,u);

  cout << y[N] << "\n";
  
}
