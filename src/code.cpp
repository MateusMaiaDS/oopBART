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

// Loglikelihood of a node
double node::loglikelihood(const Rcpp::NumericVector& residuals_values,double tau,double tau_mu){

    // Declaring quantities in the node
    int n_train = obs_train.size();
    double sum_r_sq = 0;
    double sum_r = 0;

    for(int i = 0; i<n_train;i++){
      sum_r_sq =  sum_r_sq + residuals_values(obs_train(i))*residuals_values(obs_train(i));
      sum_r = sum_r + residuals_values(obs_train(i));
    }

    return -0.5*tau*sum_r_sq -0.5*log(tau_mu+(n_train*tau)) + (0.5*(tau*tau)*(sum_r*sum_r))/( (tau*n_train)+tau_mu );
}

// Updating nu in a node;
void node::update_mu(Rcpp::NumericVector& residuals_values,double tau,double tau_mu){
  // Calculating the sum of residuals
  double sum_r = 0;
  double n_train = obs_train.size();

  for(int i = 0; i<obs_train.size();i++){
    sum_r+=residuals_values(obs_train(i));
  }

  mu = R::rnorm((sum_r*tau)/(tau*n_train+tau_mu),1/sqrt(n_train*tau+tau_mu));

}


// Getting the tree loglikelihood (BASED ON BARTMACHINES (BLEICH,2017))
void Tree::new_tree_loglike(const Rcpp::NumericVector& res_val,double tau, double tau_mu,Tree& current_tree,double& verb, int& id_node){

  if(verb < 0.25){ // Grow prob.
    t_log_likelihood = current_tree.t_log_likelihood - list_node[id_node].loglikelihood(res_val,tau,tau_mu) +
      list_node[id_node+1].loglikelihood(res_val,tau,tau_mu) +
      list_node[id_node+2].loglikelihood(res_val,tau,tau_mu);
  } else if( verb >=0.25 & verb < 0.5 ) {
    t_log_likelihood = current_tree.t_log_likelihood -
      (current_tree.list_node[id_node+1].loglikelihood(res_val,tau,tau_mu)+current_tree.list_node[id_node+2].loglikelihood(res_val,tau,tau_mu)) +
      current_tree.list_node[id_node].loglikelihood(res_val,tau,tau_mu);
  } else {
    t_log_likelihood = current_tree.t_log_likelihood -
      (current_tree.list_node[id_node+1].loglikelihood(res_val,tau,tau_mu)+current_tree.list_node[id_node+2].loglikelihood(res_val,tau,tau_mu)) +
      (list_node[id_node+1].loglikelihood(res_val,tau,tau_mu)+list_node[id_node+2].loglikelihood(res_val,tau,tau_mu));
  }

  return;
}


// Update tree all the mu from all the trees
void Tree::update_mu_tree(Rcpp::NumericVector& res_val, double tau, double tau_mu){

  // Iterating over all nodes
  for(int i = 0; i<list_node.size();i++){
    if(list_node[i].isTerminal()==1){
      list_node[i].update_mu(res_val,tau,tau_mu);
    }
  }

}

// Update tree all the mu from all the trees
void Tree::update_mu_tree_linero(Rcpp::NumericVector& res_val, double tau, double tau_mu,
                                 int& n_leaves, double& sq_mu_norm){

  // Iterating over all nodes
  for(int i = 0; i<list_node.size();i++){
    if(list_node[i].isTerminal()==1){
      list_node[i].update_mu(res_val,tau,tau_mu);
      n_leaves++;
      sq_mu_norm += list_node[i].mu*list_node[i].mu;
    }
  }

}


// Growing a tree
void Tree::grow(const Rcpp::NumericMatrix& x_train,const Rcpp::NumericMatrix& x_test,int node_min_size,const Rcpp::NumericMatrix& xcut,int &id_t,int &id_node){

  // Defining the number of covariates
  int p = x_train.ncol();
  int cov_trial_counter = 0; // To count if all the covariate were tested
  int valid_split_indicator = 0;
  int n_train,n_test;
  int g_original_index;
  int split_var;
  int max_index;
  int g_index;
  double min_x_current_node;
  double max_x_current_node;


  Rcpp::NumericVector x_cut_valid; // Defining the vector of valid "split rules" based on xcut and the terminal node
  Rcpp::NumericVector x_cut_candidates;
  node* g_node; // Node to be grow

  // Select a terminal node to be randomly selected to split
  vector<node> candidate_nodes = getTerminals();
  int n_t_node ;

  // Find a valid split indicator within a valid node
  while(valid_split_indicator==0){


    // Getting the number of terminal nodes
    n_t_node = candidate_nodes.size();

    // Getting a NumericVector of all node index
    Rcpp::NumericVector all_node_index;

    // Selecting a random terminal node
    g_index = sample_int(candidate_nodes.size());

    // Iterating over all nodes

    for(int i=0;i<list_node.size();i++){

      all_node_index.push_back(list_node[i].index); // Adding the index into the list

      if(list_node[i].index==candidate_nodes[g_index].index){
        g_original_index = i;
      }
    }


    // Getting the maximum index in case to build the new indexes for the new terminal nodes
    max_index = max(all_node_index);

    // Selecting the terminal node
    g_node = &candidate_nodes[g_index];

    // Selecting the random rule
    split_var = sample_int(p);

    // Getting the number of train
    n_train = g_node->obs_train.size();
    n_test = g_node->obs_test.size();

    // Selecting the column of the x_curr
    Rcpp::NumericVector x_current_node ;

    for(int i = 0; i<g_node->obs_train.size();i++) {
      x_current_node.push_back(x_train(g_node->obs_train(i),split_var));
    }

    min_x_current_node = min(x_current_node);
    max_x_current_node = max(x_current_node);

    // Getting available xcut variables
    x_cut_candidates = xcut(_,split_var);

    // Create a vector of splits that will lead to nontrivial terminal nodes
    for(int k = 0; k<x_cut_candidates.size();k++){
      if(x_cut_candidates(k)>min_x_current_node && x_cut_candidates(k)<max_x_current_node){
        x_cut_valid.push_back(x_cut_candidates(k));
      }
    }

    // Do not grow a tree and skip it or select a new tree
    if(x_cut_valid.size()==0){

      cov_trial_counter++;
      // CHOOSE ANOTHER SPLITING RULE
      if(cov_trial_counter == p){
        id_node = -1;
        id_t = 1;
        return;
      }
    } else {

      // Verifying that a valid split was selected
      valid_split_indicator = 1;

    }
  }

  // Sample a g_var
  double split_var_rule = x_cut_valid(sample_int(x_cut_valid.size()));

  // Creating the new terminal nodes based on the new xcut selected value
  // Getting observations that are on the left and the ones that are in the right
  Rcpp::NumericVector new_left_train_index;
  Rcpp::NumericVector new_right_train_index;
  Rcpp::NumericVector curr_obs_train; // Observations that belong to that terminal node

  // Same from above but for test observations
  // Getting observations that are on the left and the ones that are in the right
  Rcpp::NumericVector new_left_test_index;
  Rcpp::NumericVector new_right_test_index;
  Rcpp::NumericVector curr_obs_test; // Observations that belong to that terminal node

  /// Iterating over the train and test
  for(int j=0; j<n_train;j++){
    if(x_train(g_node->obs_train(j),split_var)<=split_var_rule){
      new_left_train_index.push_back(g_node->obs_train(j));
    } else {
      new_right_train_index.push_back(g_node->obs_train(j));
    }
  }

  /// Iterating over the test and test
  for(int i=0; i<n_test;i++){
    if(x_test(g_node->obs_test(i),split_var)<=split_var_rule){
      new_left_test_index.push_back(g_node->obs_test(i));
    } else {
      new_right_test_index.push_back(g_node->obs_test(i));
    }
  }

  // Updating the id_node for posterior calculations
  id_node = g_original_index;
  // Updating the current node
  list_node[g_original_index].left = max_index+1;
  list_node[g_original_index].right = max_index+2;


  // Updating the current_tree
  list_node.insert(list_node.begin()+g_original_index+1,
                   node(max_index+1,
                        new_left_train_index,
                        new_left_test_index,
                        -1,
                        -1,
                        g_node->depth+1,
                        split_var,
                        split_var_rule,
                        0));

  list_node.insert(list_node.begin()+g_original_index+2,
                   node(max_index+2,
                        new_right_train_index,
                        new_right_test_index,
                        -1,
                        -1,
                        g_node->depth+1,
                        split_var,
                        split_var_rule,
                        0));

  // Finishing the process;
  return;

}

// Pruning a tree
void Tree::prune(int& id_node){

  // Selecting possible parents of terminal nodes to prune
  vector<node> nog_list;
  Rcpp::NumericVector nog_original_index_vec;

  // Getting the parent of terminal node only
  for(int i = 0; i<list_node.size();i++){
    if(list_node[i].isTerminal()==0){
      if(list_node[i+1].isTerminal()== 1 && list_node[i+2].isTerminal()==1){
        nog_list.push_back(list_node[i]);
        nog_original_index_vec.push_back(i);
      }
    }
  }

  // Sampling a random node to be pruned
  int p_node_index = sample_int(nog_list.size());
  int nog_original_index = nog_original_index_vec(p_node_index);
  node p_node = nog_list[p_node_index];

  // Identifying which node was pruned
  id_node = nog_original_index;

  // Changing the current node that will be pruned saving their left and right indexes
  int left_pruned_index = p_node.left;
  int right_pruned_index = p_node.right;

  list_node[nog_original_index].left = -1;
  list_node[nog_original_index].right = -1;

  // Creating the list of nodes from the new tree
  vector<node> new_nodes;

  for(int i = 0; i<list_node.size();i++){

    // Adding the new nodes
    if(list_node[i].index!=left_pruned_index & list_node[i].index!=right_pruned_index){
      new_nodes.push_back(list_node[i]);
    }
  }

  // Replacing the new list of nodes;
  list_node = new_nodes;

  return;

}

// Growing a tree
void Tree::change(const Rcpp::NumericMatrix& x_train,const Rcpp::NumericMatrix& x_test,int node_min_size,const Rcpp::NumericMatrix& xcut,int& id_t,int& id_node){

  // Selecting possible parents of terminal nodes to prune
  vector<node> nog_list;
  Rcpp::NumericVector nog_original_index_vec;

  // Getting the parent of terminal node only
  for(int i = 0; i<list_node.size();i++){
    if(list_node[i].isTerminal()==0){
      if(list_node[i+1].isTerminal()== 1 && list_node[i+2].isTerminal()==1){
        nog_list.push_back(list_node[i]);
        nog_original_index_vec.push_back(i);
      }
    }
  }

  // Sampling a random node to be pruned
  node* c_node; // Node to be grow
  int p_node_index = sample_int(nog_list.size());
  int nog_original_index = nog_original_index_vec(p_node_index);
  c_node = &nog_list[p_node_index];


  // cout << "NODE CHANGED IS " << c_node->index << endl;

  // Defining the number of covariates
  int p = x_train.ncol();
  int cov_trial_counter = 0; // To count if all the covariate were tested
  int valid_split_indicator = 0;
  int split_var;
  double min_x_current_node;
  double max_x_current_node;


  Rcpp::NumericVector x_cut_valid; // Defining the vector of valid "split rules" based on xcut and the terminal node
  Rcpp::NumericVector x_cut_candidates;

  // Getting the CHANGE split rule
  while(valid_split_indicator == 0) {

    // Getting the split var
    split_var = sample_int(p);

    // Selecting the column of the x_curr
    Rcpp::NumericVector x_current_node ;

    for(int i = 0; i<c_node->obs_train.size();i++) {
      x_current_node.push_back(x_train(c_node->obs_train(i),split_var));
    }

    min_x_current_node = min(x_current_node);
    max_x_current_node = max(x_current_node);

    // Getting available xcut variables
    x_cut_candidates = xcut(_,split_var);

    // Create a vector of splits that will lead to nontrivial terminal nodes
    for(int k = 0; k<x_cut_candidates.size();k++){
      if(x_cut_candidates(k)>min_x_current_node && x_cut_candidates(k)<max_x_current_node){
        x_cut_valid.push_back(x_cut_candidates(k));
      }
    }

    // IN THIS CASE I GUESS WE WILL ALMOST NEVER GONNA GET HERE SINCE IT'S ALREADY A VALID SPLIT FROM
    //A GROW MOVE
    if(x_cut_valid.size()==0){

      cov_trial_counter++;
      // CHOOSE ANOTHER SPLITING RULE
      if(cov_trial_counter == p){
        id_node = -1;
        id_t = 1;
        return;
      }
    } else {

      // Verifying that a valid split was selected
      valid_split_indicator = 1;

    }

    // SELECTING A RANDOM SPLIT RULE AND
    double split_var_rule = x_cut_valid(sample_int(x_cut_valid.size()));

    // Creating the new terminal nodes based on the new xcut selected value
    // Getting observations that are on the left and the ones that are in the right
    Rcpp::NumericVector new_left_train_index;
    Rcpp::NumericVector new_right_train_index;
    Rcpp::NumericVector curr_obs_train; // Observations that belong to that terminal node

    // Same from above but for test observations
    // Getting observations that are on the left and the ones that are in the right
    Rcpp::NumericVector new_left_test_index;
    Rcpp::NumericVector new_right_test_index;
    Rcpp::NumericVector curr_obs_test; // Observations that belong to that terminal node

    /// Iterating over the train and test
    for(int j=0; j<c_node->obs_train.size();j++){
      if(x_train(c_node->obs_train(j),split_var)<=split_var_rule){
        new_left_train_index.push_back(c_node->obs_train(j));
      } else {
        new_right_train_index.push_back(c_node->obs_train(j));
      }
    }

    /// Iterating over the test and test
    for(int i=0; i<c_node->obs_test.size();i++){
      if(x_test(c_node->obs_test(i),split_var)<=split_var_rule){
        new_left_test_index.push_back(c_node->obs_test(i));
      } else {
        new_right_test_index.push_back(c_node->obs_test(i));
      }
    }


    // Saving the node that was selected to change
    id_node = nog_original_index;

    // Modifying the left node
    list_node[nog_original_index+1].obs_train = new_left_train_index;
    list_node[nog_original_index+1].obs_test = new_left_test_index;
    list_node[nog_original_index+1].var = split_var;
    list_node[nog_original_index+1].var_split = split_var_rule;

    // Modifying the right node
    list_node[nog_original_index+2].obs_train = new_right_train_index;
    list_node[nog_original_index+2].obs_test = new_right_test_index;
    list_node[nog_original_index+2].var = split_var;
    list_node[nog_original_index+2].var_split = split_var_rule;

    return;

  }

}

// Prior tree Loglikelihood
double Tree::prior_loglilke(double alpha, double beta){

  // Getting the val
  double p_loglike = 0;
  for(int i = 0;i<list_node.size();i++){

    // For internal nodes
    if(list_node[i].isTerminal()==0){
      p_loglike+= -beta*log(alpha*(1+list_node[i].depth));
    } else {
      p_loglike+= log(1-alpha/pow((1+list_node[i].depth),beta));
    }

  }// Finish iterating over all trees

  return p_loglike;
} // DO NOT TO CALCULATE ALL OF THEM, JUST NEED TO COMPARE WITH THE PREVIOUS TREE

// Updating tree prediction
void Tree::getPrediction(Rcpp::NumericVector &train_pred_vec,
                   Rcpp::NumericVector &test_pred_vec){

                     // Iterating over all trees nodes
                     for(int i=0; i<list_node.size();i++){

                       // Checking only over terminal nodes
                       if(list_node[i].isTerminal()==1){

                         // Iterating over the train observations
                         for(int j=0;j<list_node[i].obs_train.size();j++){
                           train_pred_vec(list_node[i].obs_train(j)) = list_node[i].mu;
                         }

                         // Iterating over the test observations
                         for(int k = 0; k<list_node[i].obs_test.size();k++){
                           test_pred_vec(list_node[i].obs_test(k)) = list_node[i].mu;
                         }

                       }

                     } // Finishing the interests over the terminal nodes

                     return;
                   }


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

void update_tau_linero(Rcpp::NumericVector y,
                         Rcpp::NumericVector y_hat,
                         double naive_sigma,
                         double& tau){

  // Transforming the variable
  int n = y.size();
  double curr_sigma,proposal_tau,proposal_sigma,acceptance,sq_residuals;

  curr_sigma = 1/sqrt(tau);

  sq_residuals = 0;
  // Calculating the residuals
  for(int i = 0; i<n;i++) {
    sq_residuals = sq_residuals + (y(i)-y_hat(i))*(y(i)-y_hat(i));
  }

  // Proposing a new tau
  proposal_tau = R::rgamma(0.5*n+1, 2/(sq_residuals));

  proposal_sigma = 1/sqrt(proposal_tau);


  // Calculating acceptance
  acceptance = exp(log(dhalf_cauchy(proposal_sigma,0,naive_sigma))+3*log(proposal_sigma)-log(dhalf_cauchy(curr_sigma,0,naive_sigma))-3*log(curr_sigma));

  // Accepting or not
  if(R::runif(0,1)<=acceptance){
    tau = proposal_tau;
  }

  return;

}

void update_tau_mu(int& n_leaves,
                   double& sq_mu_norm,
                   int& n_trees,
                   double& tau_mu){

  // Transforming the variable
  double curr_sigma_mu,proposal_tau_mu,proposal_sigma_mu,acceptance;

  curr_sigma_mu = 1/sqrt(tau_mu);

  // Proposing a new tau
  proposal_tau_mu = R::rgamma(0.5*n_leaves+1, 2/(sq_mu_norm));

  proposal_sigma_mu = 1/sqrt(proposal_tau_mu);
  cout << "Proposal tau mu is: " << proposal_tau_mu << endl;

  // Calculating acceptance
  acceptance = exp(log(dhalf_cauchy(proposal_sigma_mu,0,0.25/sqrt(n_trees)))+3*log(proposal_sigma_mu)-log(dhalf_cauchy(curr_sigma_mu,0,0.25/sqrt(n_trees)))-3*log(curr_sigma_mu));

  // Accepting or not
  if(R::runif(0,1)<=acceptance){
    tau_mu = proposal_tau_mu;
  }


  return;

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
          double a_tau, double d_tau,
          double nsigma,
          bool tau_linero,
          bool tau_mu_linero){


  // Declaring common variales
  double verb;
  double acceptance;
  double log_transition_prob_obj;
  double acceptance_ratio = 0;
  int post_counter = 0;
  int id_tree = 0; // 0 if the new tree is different from the current tree; 1 otherwise.
  int id_node = -1;
  int n_leaves;
  double sq_mu_norm;


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
  Rcpp::NumericVector tau_post,tau_mu_post;

  // Getting the initial tree
  Tree init_tree(n_train,n_test);

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
  Rcpp::NumericMatrix tree_fits_store(n_train,n_tree);

  tree_fits_store.fill(0);


  // Starting the iteration over the MCMC samples
  for(int i=0;i<n_mcmc;i++){

    // The BART sum vec
    prediction_test_sum.fill(0);
    n_leaves = 0;
    sq_mu_norm = 0 ;

    /// Iterating over the trees
    for(int t=0;t<n_tree;t++){

          // Initilaizing the id_t = 0
          id_tree = 0;

          // Updating the partial_residuals
          partial_residuals = y - partial_pred + tree_fits_store(_,t);

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
      if(tau_mu_linero){
        current_trees[t].update_mu_tree_linero(partial_residuals,tau,tau_mu,n_leaves,sq_mu_norm);
      } else {
        current_trees[t].update_mu_tree(partial_residuals,tau,tau_mu);

      }
      // Updating the predictions
      // cout << "PREDICTIION TRAIN ONE" << prediction_train(0) << endl;
      current_trees[t].getPrediction(prediction_train,prediction_test);
      // cout << "PREDICTIION TRAIN ONE" << prediction_train(0) << endl;

      // cout << " ====== " << endl;

      // Replcaing the value for partial pred
      partial_pred = partial_pred-tree_fits_store(_,t) + prediction_train;
      tree_fits_store(_,t) = prediction_train;

      // Summing up the test prediction
      prediction_test_sum += prediction_test;

      // Getting the tree size
      tree_size.push_back(current_trees[t].list_node.size());
    }

    // Updating tau
    if(tau_linero){
      update_tau_linero(y,partial_pred,nsigma,tau);
    } else {
      tau = update_tau_old(y,partial_pred,a_tau,d_tau);
    }

    // Updating tau_mu following linero equations
    if(tau_mu_linero){
      update_tau_mu(n_leaves,sq_mu_norm,n_tree,tau_mu);
    }

    // Updating the posterior matrix
    if(i >= n_burn){
      y_train_hat_post(post_counter,_) = partial_pred;
      y_test_hat_post(post_counter,_) = prediction_test_sum;

      // Updating tau
      tau_post.push_back(tau);
      tau_mu_post.push_back(tau_mu);
      post_counter++;
    }

  }

  // cout << "Acceptance Ratio = " << acceptance_ratio/n_tree << endl;

  return Rcpp::List::create(_["y_train_hat_post"] = y_train_hat_post,
                            _["y_test_hat_post"] = y_test_hat_post,
                            _["tau_post"] = tau_post,
                            _["tau_mu_post"] = tau_mu_post,
                            _["tree_size"] = tree_size);

}



//[[Rcpp::export]]
void test_grow_prune_method(Rcpp::NumericMatrix x,
                      Rcpp::NumericMatrix x_test,
                      Rcpp:: NumericVector y,
                      Rcpp:: NumericMatrix xcut,
                      double tau,
                      double tau_mu){

  Tree tree_one(y.size(),y.size());

  int id_t = 0;
  int id_node = -1;
  // Updating mu
  for(int i=0;i<100;i++){
    tree_one.grow(x,x_test,1,xcut,id_t,id_node);
  }

  // Updating mu
  for(int i=0;i<10;i++){
    tree_one.prune(id_node);
  }

  return ;
}


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


//[[Rcpp::export]]
void sum_vec(){

  Rcpp::NumericVector one, two,four;
  for(int i = 0; i <10; i++ ){
    one.push_back(i);
    two.push_back(i);
  }

  four = one + two;

  for(int i = 0; i <10; i++ ){
    cout << "Valeu of i "<< four(i) << endl;
  }

}




