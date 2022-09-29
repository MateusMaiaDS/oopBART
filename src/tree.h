#define _USE_MATH_DEFINES
#include<cmath>
#include<math.h>
#include<vector>
#include<iostream>
#include<Rcpp.h>
using namespace std;
using namespace Rcpp;

int sample_int(int n){
  return rand() % n;
}

//Half-cauchy density
double dhalf_cauchy(double x, double mu, double sigma){

  if(x>=mu){
    return (1/(M_PI_2*sigma))*(1/(1+((x-mu)*(x-mu))/(sigma*sigma)));
  } else {
    return 0.0;
  }
}

// Creating one sequence of values
Rcpp::NumericVector seq_along_cpp(int n){
  Rcpp::NumericVector vec_seq;
  for(int i =0; i<n;i++){
    vec_seq.push_back(i);
  }
  return vec_seq;
}

//

class node{

    // Storing parameters
  public:
    int index; // Storing the node index
    NumericVector obs_train; // Storing train observations index that belong to that node
    NumericVector obs_test; // Storing test observations index that belong to that node
    int left; // Index of the left node
    int right; // Index of the right node
    int parent; // Index of the parent
    int depth; //  Depth of the node

    int var; // Variable which the node was split
    double var_split; // Variable which the node was split

    double mu; // Mu parameter from the node
  public:

    // Getting the constructor
    node(int index_n, Rcpp::NumericVector obs_train_numb, Rcpp::NumericVector obs_test_numb,
         int left_i, int right_i, int depth_i, int var_i, double var_split_i,
         double mu_i){

      index = index_n;
      obs_train = obs_train_numb;
      obs_test = obs_test_numb;
      left = left_i;
      right = right_i;
      depth = depth_i;
      var = var_i;
      var_split = var_split_i;
      mu = mu_i;
    }

    void DisplayNode(){

      cout << "Node Number: " << index << endl;
      cout << "Decision Rule -> Var:  " << var << " & Rule: " << var_split << endl;
      cout << "Left <-  " << left << " & Right -> " << right << endl;

      if(true){
        cout << "Observations train: " ;
        for(int i = 0; i<obs_train.size(); i++){
          cout << obs_train[i] << " ";
        }
        cout << endl;
      }

      if(true){
        cout << "Observations test: " ;
        for(int i = 0; i<obs_test.size(); i++){
          cout << obs_test[i] << " ";
        }
        cout << endl;
      }

    }

    bool isTerminal(){
      return ((left == -1) && (right == -1) );
    }

    // Defining loglikelihood
    double loglikelihood(const Rcpp::NumericVector& residuals_values,double tau,double tau_mu);
    // Updating mu
    void update_mu(Rcpp::NumericVector& residuals_values,double tau,double tau_mu);


};


class Tree{

public:
  // Defining the main element of the tree structure
  vector<node> list_node;
  double t_log_likelihood;

  // Getting the vector of nodes
  Tree(int n_obs_train,int n_obs_test){
    // Creating a root node
    list_node.push_back(node(0,
                             seq_along_cpp(n_obs_train),
                             seq_along_cpp(n_obs_test),
                             -1, // left
                             -1, // right
                             0, //depth
                             -1, // var
                             -1.1, // var_split
                             0 )); // Mu
    // Calculating
    t_log_likelihood = 0.0;

  }


  void DisplayNodes(){
    for(int i = 0; i<list_node.size(); i++){
      list_node[i].DisplayNode();
    }
    cout << "# ====== #" << endl;
  }


  // Getting terminal nodes
  vector<node> getTerminals(){

    // Defining terminal nodes
    vector<node> terminalNodes;

    for(int i = 0; i<list_node.size(); i++){
      if(list_node[i].isTerminal()==1 ){ // Check this again, might remove the condition of being greater than 5
        terminalNodes.push_back(list_node[i]); // Adding the terminals to the list
      }
    }
    return terminalNodes;
  }

  // Getting terminal nodes
  vector<node> getNonTerminals(){

    // Defining terminal nodes
    vector<node> NonTerminalNodes;

    for(int i = 0; i<list_node.size(); i++){
      if(list_node[i].isTerminal()==0){
        NonTerminalNodes.push_back(list_node[i]); // Adding the terminals to the list
      }
    }
    return NonTerminalNodes;
  }


  // Getting the number of n_terminals
  int n_terminal(){

    // Defining the sum value
    int terminal_sum = 0;
    for(int i = 0; i<list_node.size(); i++){
      if(list_node[i].isTerminal()==1){
        terminal_sum++;
      }
    }

    return terminal_sum;
  }

  // Getting the number of non-terminals
  int n_internal(){

    // Defining the sum value
    int internal_sum = 0;
    for(int i = 0; i<list_node.size(); i++){
      if(list_node[i].isTerminal()==0){
        internal_sum++;
      }
    }

    return internal_sum;
  }

  // Get the number of NOG (branches parents of terminal nodes)
  int n_nog(){
    // Selecting possible parents of terminal nodes to prune
    vector<node> nog_list;
    int nog_counter = 0;
    // Getting the parent of terminal node only
    for(int i = 0; i<list_node.size();i++){
      if(list_node[i].isTerminal()==0){
        if(list_node[i+1].isTerminal()== 1 && list_node[i+2].isTerminal()==1){
          nog_counter++;
        }
      }
    }
    return nog_counter;
  }


  // Getting the tree loglikelihood (BASED ON BARTMACHINES (BLEICH,2017))
  void new_tree_loglike(const Rcpp::NumericVector& res_val,double tau, double tau_mu,Tree& current_tree,double& verb, int& id_node);
  // Update tree all the mu from all the trees
  void update_mu_tree(Rcpp::NumericVector& res_val, double tau, double tau_mu);
  void update_mu_tree_linero(Rcpp::NumericVector& res_val, double tau, double tau_mu,int& n_leaves, double& sq_mu_norm);
  // Growing a tree
  void grow(const Rcpp::NumericMatrix& x_train,const Rcpp::NumericMatrix& x_test,int node_min_size,const Rcpp::NumericMatrix& xcut,int &id_t,int &id_node);
  // Creating the verb to prune a tree
  void prune(int& id_node);
  // Change a tree
  void change(const Rcpp::NumericMatrix& x_train,const Rcpp::NumericMatrix& x_test,int node_min_size,const Rcpp::NumericMatrix& xcut,int& id_t,int& id_node);
  // Function to calculate the tree prior loglikelihood
  double prior_loglilke(double alpha, double beta);
  // Updating the tree predictions
  void getPrediction(Rcpp::NumericVector &train_pred_vec,Rcpp::NumericVector &test_pred_vec);

};


RCPP_EXPOSED_CLASS(node);
RCPP_EXPOSED_CLASS(Tree);

