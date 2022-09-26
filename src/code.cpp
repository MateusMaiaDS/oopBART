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


double update_tau_old(Rcpp::NumericVector y,
                      Rcpp::NumericVector y_hat,
                      double a_tau,
                      double d_tau){

  // Function used in the development of the package where I checked
  // contain_nan(y_hat);
  int n = y.size();
  double sum_sq_res = 0;
  for(int i = 0;i<n;i++){
    sum_sq_res = sum_sq_res + (y(i)-y_hat(i))*(y(i)-y_hat(i));
  }
  return R::rgamma((0.5*n+a_tau),1/(0.5*sum_sq_res+d_tau));
}

double transition_loglike(Tree& curr_tree,
                          Tree& new_tree,
                          double verb){

  // Getting the probability
  double log_prob_loglike = 0;

  // In case of Grow: (Probability to the Grew to the Current)/(From current to Grow)
  if(verb < 0.25){
    log_prob_loglike = log(0.25/new_tree.n_nog())-log(0.25/curr_tree.n_terminal());
  } else if( verb <= 0.25 && verb < 0.5) { // In case of Prune: (Prob from the Pruned to the current)/(Prob to the current to the prune)
    log_prob_loglike = log(0.25/new_tree.n_terminal()) -log(0.25/curr_tree.n_nog());
  }; // In case of change log_prob = 0; it's already the actual value

  return log_prob_loglike;

}
//[[Rcpp::export]]
List bart(const Rcpp::NumericMatrix x_train,
          const Rcpp::NumericVector y,
          const Rcpp::NumericMatrix x_test,
          const Rcpp::NumericMatrix xcut,
          int n_tree,
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
  int id_tree = 0; // 0 if the new tree is different from the current tree; 1 otherwise.
  int id_node = -1;
  // Getting the number of observations
  int n_train = x_train.rows();
  int n_test = x_test.rows();

  // Getting the tree size
  vector<int> tree_size;

  // Creating the variables
  int n_post = (n_mcmc-n_burn);
  Rcpp::NumericMatrix y_train_hat_post(n_post,n_train);
  Rcpp::NumericMatrix y_test_hat_post(n_post,n_test);

  // Getting the tau_posterior
  Rcpp::NumericVector tau_post;

  // Getting the initial tree
  Tree init_tree(n_train,n_test,alpha);

  // Calculating the initial value of loglikelihood
  init_tree.t_log_likelihood =  init_tree.list_node[0].loglikelihood(y,tau,tau_mu);

  // Creating the list of trees
  vector<Tree> current_trees;
  for(int i = 0;i<n_tree;i++){
    current_trees.push_back(init_tree);

  }

  // Creating a matrix of zeros of y_hat
  y_train_hat_post.fill(0);
  y_test_hat_post.fill(0);


  // Creating the partial residuals and partial predictions
  Rcpp::NumericVector partial_pred(n_train),partial_residuals(n_train);
  Rcpp::NumericVector prediction_train(n_train),prediction_test(n_test); // Creating the vector only for predictions
  Rcpp::NumericVector prediction_test_sum(n_test);

  // Initializing zero values
  partial_pred.fill(0);
  partial_residuals.fill(0);
  prediction_train.fill(0);
  prediction_test.fill(0);

  // Starting the iteration over the MCMC samples
  for(int i=0;i<n_mcmc;i++){

    // The BART sum vec
    prediction_test_sum.fill(0);

    /// Iterating over the trees
    for(int t=0;t<n_tree;t++){

          // Initilaizing the id_t = 0
          id_tree = 0;

          // Updating the partial_residuals
          partial_residuals = y - (partial_pred-prediction_train);

          // Creating one copy of the current tree
          Tree new_tree = current_trees[t];
          // Setting probabilities to the choice of the verb
          // Grow: 0-0.3;
          // Prune: 0.3-0.6
          // Change: 0.6-1.0
          // Swap: Not in this current implementation
          verb = R::runif(0,1);

          // Little hand to force the trees to grow
          if(current_trees[t].list_node.size()==1){
            verb = 0.1;
          }


          // Proposing a new tree given the verb
          if(verb < 0.25){
            new_tree.grow(x_train,x_test,n_min_size,xcut,id_tree,id_node);
          } else if( verb>=0.25 && verb<=0.25){
            new_tree.prune(id_node);
          } else {
            new_tree.change(x_train,x_test,n_min_size,xcut,id_tree,id_node);
          }


      if( (verb<=0.5) && (current_trees[t].list_node.size()==new_tree.list_node.size())) {
        log_transition_prob_obj = transition_loglike(current_trees[t],new_tree,verb);
      } else {
        log_transition_prob_obj = 0;
      }

      // Doing this modification only if the trees are different;
      if(id_tree==0){

            // Udating new tree loglikelihood
            new_tree.new_tree_loglike(partial_residuals,tau,tau_mu,current_trees[t],verb,id_node);
            // Getting the acceptance log value
            acceptance = new_tree.t_log_likelihood  + new_tree.prior_loglilke(alpha,beta) - current_trees[t].t_log_likelihood - current_trees[t].prior_loglilke(alpha,beta) + log_transition_prob_obj;

            // Testing if will acceptance or not
            if( (R::runif(0,1)) < exp(acceptance)){
              acceptance_ratio++;
              // cout << "ACCEPTED" << endl;
              current_trees[t] = new_tree;
            }
      } // Skipping identitical trees

      // Updating the \mu values all tree nodes;
      current_trees[t].update_mu_tree(partial_residuals,tau,tau_mu);

      // Updating the predictions
      // cout << "PREDICTIION TRAIN ONE" << prediction_train(0) << endl;
      current_trees[t].getPrediction(prediction_train,prediction_test);
      // cout << "PREDICTIION TRAIN ONE" << prediction_train(0) << endl;

      // cout << " ====== " << endl;

      // Replcaing the value for partial pred
      partial_pred = y + prediction_train - partial_residuals;

      // Summing up the test prediction
      prediction_test_sum += prediction_test;

      // Getting the tree size
      tree_size.push_back(current_trees[t].list_node.size());
    }

    // Updating tau
    tau = update_tau_old(y,partial_pred,a_tau,d_tau);

    // Updating the posterior matrix
    if(i >= n_burn){
      y_train_hat_post(post_counter,_) = partial_pred;
      y_test_hat_post(post_counter,_) = prediction_test_sum;

      // Updating tau
      tau_post.push_back(tau);
      post_counter++;
    }

  }

  // cout << "Acceptance Ratio = " << acceptance_ratio/n_tree << endl;

  return Rcpp::List::create(_["y_train_hat_post"] = y_train_hat_post,
                            _["y_test_hat_post"] = y_test_hat_post,
                            _["tau_post"] = tau_post,
                            _["tree_size"] = tree_size);

}



// //[[Rcpp::export]]
// Rcpp::NumericVector test_grow(Rcpp::NumericMatrix x,
//                       Rcpp::NumericMatrix x_test,
//                       Rcpp:: NumericVector y,
//                       Rcpp:: NumericMatrix xcut,
//                       double tau,
//                       double tau_mu){
//   Tree tree_one(y.size(),y.size());
//
//   // Updating mu
//   for(int i=0;i<5;i++){
//     tree_one.grow(x,x_test,1,xcut);
//   }
//
//   vector<node> terminal_nodes = tree_one.getTerminals();
//   Rcpp::NumericVector t_node_index(x.nrow());
//
//   for(int k = 0; k<tree_one.list_node.size();k++){
//     tree_one.list_node[k].DisplayNode();
//   }
//   for(int i = 0;i<terminal_nodes.size();i++){
//     for(int j = 0; j<terminal_nodes[i].obs_train.size();j++){
//       t_node_index(terminal_nodes[i].obs_train(j)) = terminal_nodes[i].index;
//     }
//   }
//
//   return t_node_index;
// }


// //[[Rcpp::export]]
// Rcpp::NumericVector test_prune(Rcpp::NumericMatrix x,
//                               Rcpp::NumericMatrix x_test,
//                               Rcpp:: NumericVector y,
//                               Rcpp:: NumericMatrix xcut,
//                               double tau,
//                               double tau_mu){
//   Tree tree_one(y.size(),y.size());
//
//   // Updating mu
//   for(int i=0;i<10;i++){
//     tree_one.grow(x,x_test,1,xcut);
//   }
//
//   Rcpp::NumericVector t_node_index(x.nrow());
//
//
//   cout << " =============" << endl;
//   cout << " Tree Size " << tree_one.list_node.size() << endl;
//   cout << " =============" << endl;
//
//
//   cout << " =============" << endl;
//   cout << " AFTER PRUNE " << endl;
//   cout << " =============" << endl;
//
//   for(int i = 0; i<4;i++){
//     tree_one.prune();
//   }
//
//     cout << " =============" << endl;
//     cout << " Tree Size " << tree_one.list_node.size() << endl;
//     cout << " =============" << endl;
//
//   vector<node> terminal_nodes = tree_one.getTerminals();
//
//   // Saving the index of terminal nodes
//   for(int i = 0;i<terminal_nodes.size();i++){
//     for(int j = 0; j<terminal_nodes[i].obs_train.size();j++){
//       t_node_index(terminal_nodes[i].obs_train(j)) = terminal_nodes[i].index;
//     }
//   }
//
//   return t_node_index;
// }
//
// //[[Rcpp::export]]
// Rcpp::NumericVector test_change(Rcpp::NumericMatrix x,
//                                Rcpp::NumericMatrix x_test,
//                                Rcpp:: NumericVector y,
//                                Rcpp:: NumericMatrix xcut,
//                                double tau,
//                                double tau_mu){
//   Tree tree_one(y.size(),y.size());
//
//   // Updating mu
//   for(int i=0;i<10;i++){
//     tree_one.grow(x,x_test,1,xcut);
//   }
//
//   Rcpp::NumericVector t_node_index(x.nrow());
//
//   tree_one.DisplayNodes();
//
//
//   cout << " =============" << endl;
//   cout << " Tree Size " << tree_one.list_node.size() << endl;
//   cout << " =============" << endl;
//
//
//   cout << " =============" << endl;
//   cout << " AFTER change " << endl;
//   cout << " =============" << endl;
//
//   tree_one.change(x,x_test,1,xcut);
//
//   tree_one.DisplayNodes();
//
//   cout << " =============" << endl;
//   cout << " Tree Size " << tree_one.list_node.size() << endl;
//   cout << " =============" << endl;
//
//   vector<node> terminal_nodes = tree_one.getTerminals();
//
//   // Saving the index of terminal nodes
//   for(int i = 0;i<terminal_nodes.size();i++){
//     for(int j = 0; j<terminal_nodes[i].obs_train.size();j++){
//       t_node_index(terminal_nodes[i].obs_train(j)) = terminal_nodes[i].index;
//     }
//   }
//
//   return t_node_index;
// }





