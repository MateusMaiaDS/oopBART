#define _USE_MATH_DEFINES
#include<cmath>
#include <math.h>
#include<Rcpp.h>
#include <vector>
#include "tree.h"

using namespace std;


// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]
// The line above (depends) it will make all the depe ndcies be included on the file
using namespace Rcpp;

//[[Rcpp::export]]
List bart(const Rcpp::NumericMatrix x_train,
          const Rcpp::NumericVector y,
          const Rcpp::NumericMatrix x_test,
          const Rcpp::NumericMatrix xcut,
          int n_tre,
          int n_mcmc,
          int n_burn,
          int n_min_size,
          double tau, double mu,
          double tau_mu, double naive_sigma,
          double alpha, double beta,
          double a_tau, double d_tau){


  // Declaring common variales
  double verb;
  double acceptance;
  double log_transition_prob_obj;
  double acceptance_ratio = 0;
  int post_counter = 0;
  int id_t, verb_node_index; // Boolean to check if the same tree was generated

  // Getting the number of observations
  int n_train = x_train.rows();
  int n_test = x_test.rows();

  // Creating the variables
  int n_post = (n_mcmc-n_burn);
  Rcpp::NumericMatrix y_train_hat_post(n_post,n_train);
  Rcpp::NumericMatrix y_test_hat_post(n_post,n_test);

}

//[[Rcpp::export]]
double test_likelihood(Rcpp::NumericMatrix x,
                       Rcpp:: NumericVector y,
                       double tau,
                       double tau_mu){
    Tree tree_one(y.size(),y.size());

    return tree_one.list_node[0].loglikelihood(y,tau,tau_mu);
}

//[[Rcpp::export]]
double test_mu_update(Rcpp::NumericMatrix x,
                       Rcpp:: NumericVector y,
                       double tau,
                       double tau_mu){
  Tree tree_one(y.size(),y.size());
  // Updating mu
  tree_one.list_node[0].update_mu(y,tau,tau_mu);
  return tree_one.list_node[0].mu;
}


//[[Rcpp::export]]
Rcpp::NumericVector test_grow(Rcpp::NumericMatrix x,
                      Rcpp::NumericMatrix x_test,
                      Rcpp:: NumericVector y,
                      Rcpp:: NumericMatrix xcut,
                      double tau,
                      double tau_mu){
  Tree tree_one(y.size(),y.size());

  // Updating mu
  for(int i=0;i<5;i++){
    tree_one.grow(x,x_test,1,xcut);
  }

  vector<node> terminal_nodes = tree_one.getTerminals();
  Rcpp::NumericVector t_node_index(x.nrow());

  for(int k = 0; k<tree_one.list_node.size();k++){
    tree_one.list_node[k].DisplayNode();
  }
  for(int i = 0;i<terminal_nodes.size();i++){
    for(int j = 0; j<terminal_nodes[i].obs_train.size();j++){
      t_node_index(terminal_nodes[i].obs_train(j)) = terminal_nodes[i].index;
    }
  }

  return t_node_index;
}


//[[Rcpp::export]]
Rcpp::NumericVector test_prune(Rcpp::NumericMatrix x,
                              Rcpp::NumericMatrix x_test,
                              Rcpp:: NumericVector y,
                              Rcpp:: NumericMatrix xcut,
                              double tau,
                              double tau_mu){
  Tree tree_one(y.size(),y.size());

  // Updating mu
  for(int i=0;i<10;i++){
    tree_one.grow(x,x_test,1,xcut);
  }

  Rcpp::NumericVector t_node_index(x.nrow());


  cout << " =============" << endl;
  cout << " Tree Size " << tree_one.list_node.size() << endl;
  cout << " =============" << endl;


  cout << " =============" << endl;
  cout << " AFTER PRUNE " << endl;
  cout << " =============" << endl;

  for(int i = 0; i<4;i++){
    tree_one.prune();
  }

    cout << " =============" << endl;
    cout << " Tree Size " << tree_one.list_node.size() << endl;
    cout << " =============" << endl;

  vector<node> terminal_nodes = tree_one.getTerminals();

  // Saving the index of terminal nodes
  for(int i = 0;i<terminal_nodes.size();i++){
    for(int j = 0; j<terminal_nodes[i].obs_train.size();j++){
      t_node_index(terminal_nodes[i].obs_train(j)) = terminal_nodes[i].index;
    }
  }

  return t_node_index;
}

//[[Rcpp::export]]
Rcpp::NumericVector test_change(Rcpp::NumericMatrix x,
                               Rcpp::NumericMatrix x_test,
                               Rcpp:: NumericVector y,
                               Rcpp:: NumericMatrix xcut,
                               double tau,
                               double tau_mu){
  Tree tree_one(y.size(),y.size());

  // Updating mu
  for(int i=0;i<10;i++){
    tree_one.grow(x,x_test,1,xcut);
  }

  Rcpp::NumericVector t_node_index(x.nrow());

  tree_one.DisplayNodes();


  cout << " =============" << endl;
  cout << " Tree Size " << tree_one.list_node.size() << endl;
  cout << " =============" << endl;


  cout << " =============" << endl;
  cout << " AFTER change " << endl;
  cout << " =============" << endl;

  tree_one.change(x,x_test,1,xcut);

  tree_one.DisplayNodes();

  cout << " =============" << endl;
  cout << " Tree Size " << tree_one.list_node.size() << endl;
  cout << " =============" << endl;

  vector<node> terminal_nodes = tree_one.getTerminals();

  // Saving the index of terminal nodes
  for(int i = 0;i<terminal_nodes.size();i++){
    for(int j = 0; j<terminal_nodes[i].obs_train.size();j++){
      t_node_index(terminal_nodes[i].obs_train(j)) = terminal_nodes[i].index;
    }
  }

  return t_node_index;
}

//[[Rcpp::export]]
double using_member(int x){
  test t(x);
  return t.sum_double_x();
}




