#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <time.h>
#include "SoftTree.cpp"

void readFromFile(string filename, vector< vector<double> > &X,
                  vector<double> &Y) {
  ifstream file;
  file.open(filename.c_str());
  assert(file.is_open());
  
  double val;
  vector<double> row;
  
  while(!file.eof())
  {
    file >> val;
    if (file.peek() == '\n') { // if line ends, 'val' is the response value
      Y.push_back(val);
      X.push_back(row);
      row.clear();
    } else {                  // else, it is an entry of the feature vector
      row.push_back(val);
    }
  }
}


int main()
{
  

  vector< vector< double> > X, V, U;
  vector<double> Y, R, T;

  //srand(time(NULL)); // random seed
  srand(123457);

  cout.precision(5);  // # digits after decimal pt
  cout.setf(ios::fixed,ios::floatfield);
    
    int iter = 0;
    string dataset = "abalone";
 
  string filename;
    
    string prefix = "STdata/train_";
  filename = prefix
           + (char)(iter+'0')
           + '_'
           + dataset
           + ".txt";
  
  readFromFile(filename, X, Y);
 
    
prefix = "STdata/val_";
  filename = prefix
           + (char)(iter+'0')
           + '_'
           + dataset
           + ".txt";
  
  
  readFromFile(filename, V, R);
  
  
    
prefix = "STdata/test_";
  filename = prefix
           + (char)(iter+'0')
           + '_'
           + dataset
           + ".txt";
    
  readFromFile(filename, U, T);
    

  clock_t time_start = clock() ;
    
    
  SoftTree st(X, Y, V, R);
  
  double y;
  double mse=0;
  
  //ofstream outf("out");
  
  mse = st.meanSqErr(X, Y);
  

  mse = st.meanSqErr(U, T);
  
  cout << "test_error: " << mse << "\t";
  cout << endl;
    
    
  clock_t time_end = clock() ;
    
  double duration = double(time_end - time_start)/CLOCKS_PER_SEC;
    
    cout<< "time" << duration;
    
  return 0;
}
